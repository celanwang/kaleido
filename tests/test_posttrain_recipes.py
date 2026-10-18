import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from kaleido.demos.post_training import Config, execute  # noqa:E402
from kaleido.posttrain.fixture import load_fixture  # noqa:E402
from kaleido.posttrain.model import (  # noqa:E402
    inject_lora,
    pack_completions,
    read_adapter,
    sft_tokens,
)
from kaleido.posttrain.recipes import accepted_completions, decode_completion, train_rejection_sft  # noqa:E402
from kaleido.posttrain.tasks import make_tasks, training_reward  # noqa:E402

ROOT = Path(__file__).resolve().parents[1]


def smoke_config():
    return Config(**json.loads((ROOT / "configs/p4_smoke.json").read_text()))


def test_offline_fixture_end_to_end_and_explicit_no_acceptance(tmp_path):
    path, metrics = execute(smoke_config(), tmp_path, root=tmp_path, fixture=True)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["config"]["experiment_kind"] == "fixture"
    assert (
        json.loads((path / "model_source.json").read_text())["kind"]
        == "random_fixture_not_pretrained"
    )
    assert metrics["3"]["grpo"]["training"]["updates"] == 2
    assert metrics["3"]["rejection_sft"]["training"]["status"] == "no_accepted_completions"
    checks = json.loads((path / "checks.json").read_text())["3"]
    assert checks["base_weights_frozen"] and checks["shared_prompt_schedule"]
    for method in ("sft", "grpo", "rejection"):
        arrays, metadata = read_adapter(path / f"{method}-3.npz")
        assert arrays and metadata["experiment_kind"] == "fixture"
    histories = json.loads((path / "histories.json").read_text())["3"]
    assert (
        metrics["3"]["sft"]["training"]["selected_step"]
        == max(
            histories["sft"],
            key=lambda row: (
                row["validation"]["strict_accuracy"],
                -row["validation"]["response_nll"],
            ),
        )["step"]
    )


def test_fixture_seeded_replay(tmp_path):
    a, first = execute(smoke_config(), tmp_path, root=tmp_path, fixture=True)
    b, second = execute(smoke_config(), tmp_path, root=tmp_path, fixture=True)
    assert first == second
    assert json.loads((a / "test_samples.json").read_text()) == json.loads(
        (b / "test_samples.json").read_text()
    )


def test_rejection_filter_uses_only_unique_verified_generated_ids(tmp_path, monkeypatch):
    from kaleido.posttrain import recipes

    model, tokenizer, _ = load_fixture()
    inject_lora(model, rank=2, alpha=4, seed=3)
    tasks = make_tasks(train_per_family=2, heldout_per_family=1)
    train = [r for r in tasks if r["split"] == "train"]
    valid = [r for r in tasks if r["split"] == "valid"]

    def successful(model, tokenizer, rows, *, group_size, max_new_tokens, seed):
        expanded = [r for r in rows for _ in range(group_size)]
        pairs = [sft_tokens(tokenizer, r) for r in expanded]
        batch = pack_completions([p[0] for p in pairs], [p[1] for p in pairs], pad_id=0)
        records = [
            {"id": r["id"], "completion_ids": pair[1], "training_reward": 1.0}
            for r, pair in zip(expanded, pairs, strict=True)
        ]
        return batch, [1.0] * len(expanded), records

    monkeypatch.setattr(recipes, "collect_rollout", successful)
    result, _, rollouts, _ = train_rejection_sft(
        model,
        tokenizer,
        train,
        valid,
        iterations=2,
        prompts_per_iteration=1,
        group_size=2,
        updates=2,
        batch_size=2,
        eval_interval=2,
        max_new_tokens=2,
    )
    assert (
        result["status"] == "completed"
        and result["updates"] == 2
        and result["update_completion_tokens"] > 0
    )
    assert result["accepted_unique"] == len(accepted_completions(rollouts))
    assert result["accepted_raw"] == 4 and result["accepted_unique"] <= 2


def test_special_token_injection_is_not_hidden_by_decoding():
    model, tokenizer, _ = load_fixture()
    row = make_tasks(train_per_family=1, heldout_per_family=1)[0]
    ids = (
        tokenizer.encode("<answer>")
        + [1]
        + tokenizer.encode(str(row["answer"]) + "</answer>")
        + [2]
    )
    response = decode_completion(model, tokenizer, ids)
    assert "<special:1>" in response and training_reward(row, response) == 0


@pytest.mark.parametrize(
    "options",
    [
        {"seeds": []},
        {"group_size": 1},
        {"sft_steps": 0},
        {"post_lr": -1},
        {"device": "cuda"},
        {"beta": float("nan")},
    ],
)
def test_config_boundaries(options):
    with pytest.raises(ValueError):
        Config(**options)
