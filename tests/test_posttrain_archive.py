"""Audit real P4 outcomes offline; do not download the 0.5B base."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kaleido.posttrain.model import prompt_tokens, read_adapter, sft_tokens  # noqa:E402
from kaleido.posttrain.recipes import accepted_completions  # noqa:E402
from kaleido.posttrain.tasks import (  # noqa:E402
    dataset_digest,
    make_tasks,
    score_response,
    summarize,
    training_reward,
)

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/p4/reference"


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    jsons = {p.stem: json.loads(p.read_text()) for p in FOLDER.glob("*.json")}
    directory = tmp_path_factory.mktemp("native-qwen-tokenizer")
    index = json.loads((FOLDER / "tokenizer/files.json").read_text())
    for name, identity in index.items():
        assert Path(name).name == name
        raw = gzip.decompress((FOLDER / "tokenizer" / (name + ".gz")).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == identity["sha256"]
        (directory / name).write_bytes(raw)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    return jsons, tokenizer


def test_frozen_native_identity_tasks_and_template(archive):
    jsons, tokenizer = archive
    config = jsons["manifest"]["config"]
    assert config["experiment_kind"] == "formal_qwen" and jsons["manifest"]["status"] == "completed"
    assert jsons["manifest"]["environment"]["git_dirty"] is False
    source = jsons["model_source"]
    assert source["repo_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert source["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert (
        hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
        == source["chat_template_sha256"]
    )
    for name, stored in (
        ("LICENSE", "QWEN-LICENSE.txt"),
        ("config.json", "base_config.json"),
        ("generation_config.json", "base_generation_config.json"),
    ):
        assert hashlib.sha256((FOLDER / stored).read_bytes()).hexdigest() == source["files"][name]
    rows = make_tasks(
        seed=config["task_seed"],
        train_per_family=config["train_per_family"],
        heldout_per_family=config["heldout_per_family"],
        operand_max=config["operand_max"],
    )
    assert (
        rows == jsons["tasks"]["rows"] and dataset_digest(rows) == jsons["tasks"]["dataset_sha256"]
    )
    for row in rows:
        assert prompt_tokens(tokenizer, row) == jsons["prompts"][row["id"]]
    assert jsons["resources"]["base_parameters"] == 494032768
    assert (
        jsons["resources"]["adapter_parameters"] == 540672
        and jsons["resources"]["native_greedy_match"]
    )


def test_every_evaluation_decodes_and_recomputes_independently(archive):
    jsons, tokenizer = archive
    lookup = {r["id"]: r for r in jsons["tasks"]["rows"]}
    eos = jsons["base_generation_config"]["eos_token_id"]

    def verify(records):
        for sample in records:
            ids = sample["completion_ids"]
            content = ids[:-1] if ids and ids[-1] in eos else ids
            response = tokenizer.decode(content, skip_special_tokens=False)
            assert response == sample["response"]
            expected = score_response(lookup[sample["id"]], response, truncated=sample["truncated"])
            for name, value in expected.items():
                assert value == sample[name]

    for group in jsons["base_evaluation"].values():
        verify(group["samples"])
        computed = summarize(group["samples"])
        for name, value in computed.items():
            assert value == group["metrics"][name]
    for seed, methods in jsons["test_samples"].items():
        for method, records in methods.items():
            verify(records)
            computed = summarize(records)
            for name, value in computed.items():
                assert value == jsons["metrics"][seed][method]["test"][name]
            assert {lookup[r["id"]]["split"] for r in records} == {"test"}
        for evaluations in jsons["validation_samples"][seed].values():
            for evaluation in evaluations:
                verify(evaluation["samples"])
                assert {lookup[r["id"]]["split"] for r in evaluation["samples"]} == {"valid"}
    assert jsons["baselines"]["evaluation"]["test"]["strict_accuracy"] == 0.125


def test_frozen_group_signals_matched_schedules_and_all_token_budgets(archive):
    jsons, tokenizer = archive
    config = jsons["manifest"]["config"]
    size = config["group_size"]
    train = [r for r in jsons["tasks"]["rows"] if r["split"] == "train"]
    lookup = {r["id"]: r for r in train}
    for seed, methods in jsons["rollouts"].items():
        schedules = []
        for method, iterations in methods.items():
            assert len(iterations) == config["iterations"]
            schedules.append([r["question_ids"] for r in iterations])
            generated = 0
            zero_groups = 0
            for iteration in iterations:
                records = iteration["samples"]
                assert len(records) == size * config["prompts_per_iteration"]
                generated += sum(len(s["completion_ids"]) for s in records)
                group_counts = Counter()
                for start in range(0, len(records), size):
                    group = records[start : start + size]
                    assert len({s["id"] for s in group}) == 1 and group[0]["id"] in lookup
                    rewards = [
                        training_reward(lookup[s["id"]], s["response"], truncated=s["truncated"])
                        for s in group
                    ]
                    assert rewards == [s["training_reward"] for s in group]
                    category = (
                        "all_wrong"
                        if sum(rewards) == 0
                        else "all_correct"
                        if sum(rewards) == size
                        else "mixed"
                    )
                    group_counts[category] += 1
                    zero_groups += int(category != "mixed")
                if method == "grpo":
                    expected = (group_counts["all_wrong"] + group_counts["all_correct"]) / config[
                        "prompts_per_iteration"
                    ]
                    assert iteration["group_stats"]["zero_variance_group_fraction"] == expected
            training = jsons["metrics"][seed][method]["training"]
            assert training["generated_completion_tokens"] == generated
            if method == "grpo":
                assert training["update_completion_tokens"] == generated * config["grpo_epochs"]
                assert training["zero_variance_group_fraction"] == pytest.approx(
                    zero_groups / (config["iterations"] * config["prompts_per_iteration"])
                )
            else:
                accepted = accepted_completions(iterations)
                assert len(accepted) == training["accepted_unique"]
                rng = np.random.default_rng(int(seed) + 400)
                count = 0
                for _ in range(training["updates"]):
                    ids = rng.integers(
                        0, len(accepted), size=config["prompts_per_iteration"] * size
                    )
                    count += sum(len(accepted[int(i)]["completion_ids"]) for i in ids)
                assert count == training["update_completion_tokens"]
        assert schedules[0] == schedules[1]
        rng = np.random.default_rng(int(seed) + 100)
        count = 0
        for _ in range(config["sft_steps"]):
            indices = rng.choice(
                len(train), size=min(config["sft_batch_size"], len(train)), replace=False
            )
            count += sum(len(sft_tokens(tokenizer, train[int(i)])[1]) for i in indices)
        assert count == jsons["metrics"][seed]["sft"]["training"]["update_completion_tokens"]


@pytest.mark.parametrize("seed", [42, 7, 2026])
def test_adapter_identity_reference_and_validation_selection(archive, seed):
    jsons, _ = archive
    key = str(seed)
    sft, sft_meta = read_adapter(FOLDER / f"sft-{seed}.npz")
    assert any(np.any(v != 0) for name, v in sft.items() if name.endswith("lora_b"))
    reference = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in sft.items()}
    for method, filename, position in (
        ("sft", "sft", "step"),
        ("grpo", "grpo", "iteration"),
        ("rejection_sft", "rejection", "step"),
    ):
        state, metadata = read_adapter(FOLDER / f"{filename}-{seed}.npz")
        assert (
            metadata["experiment_kind"] == "formal_qwen"
            and metadata["model_source"] == sft_meta["model_source"]
        )
        history = [r for r in jsons["histories"][key][method] if "validation" in r]
        best = max(
            history,
            key=lambda row: (
                row["validation"]["strict_accuracy"],
                -row["validation"]["response_nll"],
            ),
        )
        assert metadata["selection"] == best[position]
        if method != "sft":
            assert metadata["reference_sft_arrays"] == reference
            if metadata["selection"] == 0:
                for name, value in state.items():
                    np.testing.assert_array_equal(value, sft[name])
    assert (
        jsons["checks"][key]["base_weights_frozen"]
        and jsons["checks"][key]["shared_prompt_schedule"]
    )
    replay = [r for r in jsons["replay"]["items"] if r["seed"] == seed]
    assert len(replay) == 3 and all(r["token_ids_match"] and r["score_matches"] for r in replay)
    assert jsons["replay"]["source_sha256"] == jsons["manifest"]["environment"]["source_sha256"]
    for item in replay:
        assert (
            item["strict_accuracy"]
            == jsons["metrics"][key][item["method"]]["test"]["strict_accuracy"]
        )
