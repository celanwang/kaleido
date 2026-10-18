import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from kaleido.demos.model_evaluation import Config, archived_panel, execute  # noqa: E402
from kaleido.evaluation.adapters import QwenAdapter  # noqa: E402
from kaleido.evaluation.comparison import compare  # noqa: E402
from kaleido.evaluation.schema import Case, GenerationProtocol  # noqa: E402
from kaleido.posttrain.fixture import load_fixture  # noqa: E402
from kaleido.posttrain.model import adapter_state, inject_lora, load_adapter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_shared_qwen_adapter_reloads_own_weights_and_does_not_send_gold():
    model, tokenizer, _ = load_fixture()
    inject_lora(model, rank=2, alpha=4)
    state = adapter_state(model)
    adapter = QwenAdapter("a", model, tokenizer, state, source={"fixture": True})
    cases = [Case("q", "s", "f", "What is two plus two?", "GOLD_SENTINEL")]
    prompts = []
    original = tokenizer.apply_chat_template

    def traced(messages, **kwargs):
        prompts.append(messages)
        return original(messages, **kwargs)

    tokenizer.apply_chat_template = traced
    first = adapter.generate(cases, GenerationProtocol(max_new_tokens=3))
    changed = {k: v + 0.25 for k, v in state.items()}
    load_adapter(model, changed)
    second = adapter.generate(cases, GenerationProtocol(max_new_tokens=3))
    assert first == second
    for key, value in adapter_state(model).items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    assert "GOLD_SENTINEL" not in json.dumps(prompts)


def test_comparison_reproduces_P4_and_rejects_protocol_mismatch():
    panel = archived_panel(ROOT, (42, 7, 2026))
    stats = compare(panel["results"], (42, 7, 2026), resamples=100)
    key = "p4_original_test:grpo:2026"
    assert stats["tests"][key]["wins"] == 5
    assert stats["tests"][key]["losses"] == 1
    assert stats["tests"][key]["holm_p"] == 1
    panel["results"]["grpo-42"]["identity"]["protocol_sha256"] = "other"
    with pytest.raises(ValueError):
        compare(panel["results"], (42, 7, 2026), resamples=100)


def test_fixture_full_pipeline_is_labeled_and_replays_real_P3(tmp_path):
    config = Config(**json.loads((ROOT / "configs/p5_smoke.json").read_text()))
    path = execute(config, tmp_path, fixture=True)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["config"]["fixture"]
    checks = json.loads((path / "checks.json").read_text())
    assert checks["no_training"] and checks["p4_scoring_reproduced"]
    losses = np.load(path / "p3_token_losses.npz", allow_pickle=False)["42"]
    original = np.load(ROOT / "results/p3/reference/evaluation.npz", allow_pickle=False)
    assert len(losses) > 10000
    # Match the original P3 token trace independent of device's floating-point reduction.
    np.testing.assert_allclose(losses, original["42__test_nll"], atol=1e-5, rtol=1e-5)
    judge = json.loads((path / "judge_calibration.json").read_text())
    assert not judge["enabled"]
    assert json.loads((path / "judge_panel.json").read_text())["status"] == "disabled"
