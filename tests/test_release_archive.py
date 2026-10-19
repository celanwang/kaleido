"""Audit P6 replay and actual deployment evidence without model downloads."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from kaleido.experiments import array_digest
from kaleido.repro.integrity import file_info, verify_index

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/p6/reference"


def load(name):
    return json.loads((FOLDER / (name + ".json")).read_text())


def test_release_evidence_checksums_clean_sources_and_all_six_formal_recipes():
    index = load("artifacts")
    assert set(index["files"]) == {
        str(p.relative_to(FOLDER))
        for p in FOLDER.rglob("*")
        if p.is_file() and p != FOLDER / "artifacts.json"
    }
    checked = verify_index(FOLDER, index, deep=True)
    assert checked["ok"] and checked["checked_arrays"] > 80
    reproduction = load("reproduction")
    assert set(reproduction["stages"]) == {f"p{i}" for i in range(6)}
    assert reproduction["suite_manifest"]["status"] == "completed"
    for value in reproduction["stages"].values():
        assert value["receipt"]["profile"] == "formal" and not value["receipt"]["fixture"]
        assert value["manifest"]["status"] == "completed"
        assert value["manifest"]["environment"]["git_dirty"] is False
        assert value["manifest"]["environment"]["source_hash_version"] == 2
        assert value["manifest"]["environment"]["git_revision"].startswith("50ebad6")
        config = Path(value["receipt"]["config_file"]).name
        assert file_info(ROOT / "configs" / config) == value["receipt"]["config_identity"]
    smoke = load("smoke")
    assert smoke["manifest"]["config"]["profile"] == "smoke"
    assert len(smoke["steps"]["completed"]) == 6
    assert all(r["fixture"] for r in smoke["steps"]["completed"] if r["stage"] in ("p4", "p5"))


def test_metric_comparisons_and_nonidentical_training_are_not_silently_passed():
    reproduction = load("reproduction")
    compare = runpy.run_path(str(ROOT / "scripts/summarize_reproduction.py"))["differences"]
    for stage, value in reproduction["stages"].items():
        for name, computed in value["json_comparisons"].items():
            old = json.loads((ROOT / "results" / stage / "reference" / name).read_text())
            assert compare(old, computed["actual"]) == computed["differences"]
    assert not reproduction["all_compared_arrays_identical"]
    assert not reproduction["all_compared_metrics_match"]
    diagnostics = load("training_diagnostics")
    assert diagnostics["hash_field_differences"] == 84
    assert diagnostics["performance_field_differences"] == 6
    assert diagnostics["other_task_metric_differences"] == []
    for seed, values in diagnostics["p3"].items():
        new = FOLDER / f"changed/p3/model-{seed}.npz"
        old = ROOT / f"results/p3/reference/model-{seed}.npz"
        with np.load(new, allow_pickle=False) as a, np.load(old, allow_pickle=False) as b:
            error = max(float(np.abs(a[k] - b[k]).max()) for k in a)
            assert error == values["max_abs_weight_error"] > 0
            hashes = {k: array_digest(a[k]) for k in a}
            assert (
                hashes == reproduction["stages"]["p3"]["array_comparisons"][new.name]["new_digests"]
            )
        assert abs(values["delta_test_nll"]) < 1e-5 and values["initialization_sha256_equal"]


def test_frozen_traces_and_p4_adapters_reproduce_without_overwriting_history():
    reproduction = load("reproduction")
    assert reproduction["all_compared_traces_identical"]
    p4 = reproduction["stages"]["p4"]
    assert len(p4["array_comparisons"]) == 9
    assert all(v["identical"] for v in p4["array_comparisons"].values())
    assert p4["trace_comparisons"]["test_samples.json"]["identical"]
    assert reproduction["stages"]["p5"]["trace_comparisons"]["evaluations.json"]["identical"]
    old = json.loads((ROOT / "results/p3/reference/generations.json").read_text())
    actual = load("p3_generations")
    assert (old == actual) == load("training_diagnostics")["p3_generations_identical"]
    state = load("release_state")
    assert state["project_version"] == "1.0.0"
    assert not state["github_release_published"] and not state["draft_workflow_executed"]


def test_published_bundle_matches_original_trained_model_and_independent_wheel_record():
    torch = pytest.importorskip("torch")
    from kaleido.deployment.runtime import Predictor
    from kaleido.lm.training import load_weights

    predictor = Predictor(FOLDER / "bundle")
    assert predictor.manifest["source"]["environment"]["git_dirty"] is False
    assert predictor.manifest["source"]["weights"] == file_info(
        ROOT / "results/p3/reference/model-42.npz"
    )
    eager = load_weights(ROOT / "results/p3/reference/model-42.npz")
    for length in (1, 128):
        ids = list(range(length))
        with torch.inference_mode():
            expected = eager(torch.tensor([ids]))[0]
        torch.testing.assert_close(predictor.logits(ids), expected, atol=1e-5, rtol=1e-5)
    examples = json.loads((FOLDER / "bundle/examples.json").read_text())["cases"]
    proof = load("wheel_check")
    assert proof["status"] == "completed" and proof["exact_token_replay"]
    assert (
        proof["installation"]["version"] == "1.0.0"
        and not proof["installation"]["source_on_sys_path"]
    )
    assert proof["offline_install"] and proof["no_training_imports"] and proof["outside_checkout"]
    assert proof["runtime_lockfile"] == file_info(ROOT / "uv.lock")
    assert not proof["installation"]["hf_packages_required"]
    for example, result in zip(examples, proof["requests"], strict=True):
        assert predictor.predict(example["request"])["completion_ids"] == result["completion_ids"]
