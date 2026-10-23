"""Audit P5 evidence offline; no official base weight download is needed."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kaleido.evaluation.benchmarks import build_suite
from kaleido.evaluation.comparison import compare
from kaleido.evaluation.judge import GATES, judge_cases, summarize_judgments
from kaleido.evaluation.schema import Case, digest, freeze_cases
from kaleido.evaluation.scoring import score, stratified_summary
from kaleido.evaluation.statistics import document_nll_bootstrap

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/p5/reference"


@pytest.fixture(scope="module")
def evidence():
    return {p.stem: json.loads(p.read_text()) for p in FOLDER.glob("*.json")}


def test_frozen_identity_benchmarks_and_all_file_checksums(evidence):
    manifest = evidence["manifest"]
    assert manifest["status"] == "completed" and not manifest["config"]["fixture"]
    assert manifest["environment"]["git_dirty"] is False
    assert manifest["environment"]["git_revision"].startswith("9e87760")
    cases, frozen = build_suite(ROOT)
    assert evidence["benchmarks"] == {**frozen, "cases": [c.to_dict() for c in cases]}
    assert freeze_cases(cases) == frozen["case_sha256"]
    assert len(evidence["evaluations"]) == 10
    index = evidence["artifacts"]["files"]
    assert set(index) == {p.name for p in FOLDER.iterdir() if p.name != "artifacts.json"}
    for name, info in index.items():
        raw = (FOLDER / name).read_bytes()
        assert len(raw) == info["bytes"] and hashlib.sha256(raw).hexdigest() == info["sha256"]


def test_all_1080_rows_rescore_and_native_token_traces_decode(evidence, tmp_path):
    transformers = pytest.importorskip("transformers")
    directory = ROOT / "results/p4/reference/tokenizer"
    for name, info in json.loads((directory / "files.json").read_text()).items():
        raw = gzip.decompress((directory / (name + ".gz")).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == info["sha256"]
        (tmp_path / name).write_bytes(raw)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tmp_path, local_files_only=True, trust_remote_code=False
    )
    eos = json.loads((ROOT / "results/p4/reference/base_generation_config.json").read_text())[
        "eos_token_id"
    ]
    cases = {r["id"]: Case(**r) for r in evidence["benchmarks"]["cases"]}
    count = 0
    for value in evidence["evaluations"].values():
        assert value["identity_sha256"] == digest(value["identity"])
        assert value["identity"]["case_sha256"] == freeze_cases(list(cases.values()))
        assert value["summary"] == stratified_summary(value["records"])
        assert {r["id"] for r in value["records"]} == set(cases)
        for r in value["records"]:
            case = cases[r["id"]]
            expected = score(case, r["response"], truncated=r["truncated"])
            assert all(r[k] == v for k, v in expected.items())
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": case.system},
                    {"role": "user", "content": case.prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            assert prompt == r["prompt_ids"] and len(prompt) == r["prompt_tokens"]
            ids = r["completion_ids"]
            assert len(ids) == r["generated_tokens"] <= 64
            assert r["truncated"] == (ids[-1] not in eos)
            content = ids[:-1] if ids[-1] in eos else ids
            assert tokenizer.decode(content, skip_special_tokens=False) == r["response"]
            count += 1
    assert count == 1080


def test_paired_statistics_and_judge_gate_recompute(evidence):
    seeds = evidence["manifest"]["config"]["seeds"]
    assert compare(evidence["evaluations"], seeds) == evidence["comparisons"]
    assert compare(evidence["p4_archive"]["results"], seeds) == evidence["p4_comparisons"]
    assert len(evidence["comparisons"]["tests"]) == 18
    for key, value in evidence["comparisons"]["conditional_intervals"].items():
        assert value["fixed_seeds"] == 3
        assert value["questions"] == (
            64 if key.startswith("novel") else 32 if key.startswith("gsm") else 12
        )
    judge = evidence["judge_calibration"]
    assert judge["thresholds"] == GATES
    for split in ("calibration", "audit"):
        saved = judge[split]
        recomputed = summarize_judgments(saved["pairs"], saved["records"])
        assert recomputed == {"summary": saved["summary"], "records": saved["records"]}
        assert len(judge_cases(saved["pairs"])) == 32
    assert not judge["enabled"] and evidence["judge_panel"]["status"] == "disabled"


def test_real_P3_token_losses_and_document_weighting(evidence):
    current = np.load(FOLDER / "p3_token_losses.npz", allow_pickle=False)
    original = np.load(ROOT / "results/p3/reference/evaluation.npz", allow_pickle=False)
    for seed, value in evidence["p3_likelihood"]["models"].items():
        losses = current[seed]
        np.testing.assert_allclose(losses, original[f"{seed}__test_nll"], atol=1e-5, rtol=1e-5)
        assert value["metrics"]["nll"] == pytest.approx(losses.mean())
        assert value["metrics"]["target_tokens"] == len(losses)
        assert sum(r["tokens"] for r in value["documents"]) == len(losses)
        assert sum(r["nll_sum"] for r in value["documents"]) == pytest.approx(losses.sum())
        assert document_nll_bootstrap(value["documents"]) == value["uncertainty"]
        path = ROOT / f"results/p3/reference/model-{seed}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value["weights_file_sha256"]


def test_protocol_repair_preserves_the_original_evidence(evidence):
    old = json.loads((ROOT / "results/p5/protocol-audit/context-v1.json").read_text())
    assert old["status"] == "superseded_protocol_defect_not_model_ranking"
    new = [c for c in evidence["benchmarks"]["cases"] if c["suite"] == "context_qa_v2"]
    assert [(c["prompt"], c["answer"]) for c in old["cases"]] == [
        (c["prompt"], c["answer"]) for c in new
    ]
    assert all("exactly one color word" in c["system"] for c in new)
    assert evidence["replay"]["unchanged_numerical_records"] == 960
    assert evidence["replay"]["all_numerical_records_identical"]
    assert evidence["replay"]["all_judge_records_identical"]
