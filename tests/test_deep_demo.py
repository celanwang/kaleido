import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from kaleido.core import Tensor  # noqa:E402
from kaleido.core.models import DigitCNN, GaussianVAE, MemoryClassifier  # noqa:E402
from kaleido.core.nn import MLP  # noqa:E402
from kaleido.demos.deep_learning import Config, _vae_metrics, execute  # noqa:E402
from kaleido.experiments import array_digest  # noqa:E402


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("p2")
    path, metrics = execute(Config(seeds=(42,)), root, root=root)
    jsons = {p.stem: json.loads(p.read_text()) for p in path.glob("*.json")}
    arrays = {
        name: dict(np.load(path / (name + ".npz"), allow_pickle=False))
        for name in ("data", "models", "outputs")
    }
    return path, metrics, jsons, arrays


def test_dataset_fingerprints_splits_and_checkpoint_selection(experiment):
    _, metrics, jsons, arrays = experiment
    assert jsons["manifest"]["status"] == "completed"
    for kind in ("data", "models", "outputs"):
        for name, array in arrays[kind].items():
            assert array_digest(array) == jsons[kind]["array_digests"][name]
    split = jsons["data"]["splits"]["42"]["splits"]
    assert sorted(sum(split.values(), [])) == list(range(len(arrays["data"]["digits_y"])))
    for method in ("mlp_sgd", "mlp_adam", "cnn"):
        history = jsons["histories"]["42"][method]
        best = min(history, key=lambda row: row["valid_nll"])
        assert best["epoch"] == metrics["42"][method]["selected_epoch"]
        assert metrics["42"][method]["test_accuracy"] > 0.90
    for method in ("vae", "vae_no_kl"):
        best = min(jsons["histories"]["42"][method], key=lambda row: row["negative_elbo"])
        assert best["epoch"] == metrics["42"][method]["selected_epoch"]


def test_serialized_models_reproduce_predictions_and_metrics(experiment):
    _, metrics, _, arrays = experiment

    def state(prefix):
        return {
            name.removeprefix(prefix): value
            for name, value in arrays["models"].items()
            if name.startswith(prefix)
        }

    ids = arrays["outputs"]["42__test_ids"]
    x = arrays["data"]["digits_x"][ids]
    y = arrays["data"]["digits_y"][ids]
    for name in ("mlp_sgd", "mlp_adam", "cnn"):
        model = (
            DigitCNN(rng=np.random.default_rng(0))
            if name == "cnn"
            else MLP([64, 32, 10], rng=np.random.default_rng(0))
        )
        model.load_state_dict(state("42__" + name + "__"))
        prediction = model(Tensor(x.reshape(-1, 1, 8, 8) if name == "cnn" else x)).data
        np.testing.assert_allclose(
            prediction, arrays["outputs"]["42__" + name + "_logits"], atol=1e-12
        )
        assert np.mean(prediction.argmax(1) == y) == metrics["42"][name]["test_accuracy"]
    vae = GaussianVAE(64, 16, 4, rng=np.random.default_rng(0))
    vae.load_state_dict(state("42__vae__"))
    observed, reconstructed = _vae_metrics(vae, x, arrays["outputs"]["42__vae_test_epsilon"])
    np.testing.assert_allclose(
        reconstructed, arrays["outputs"]["42__vae_reconstruction"], atol=1e-12
    )
    for key in observed:
        assert observed[key] == pytest.approx(metrics["42"]["vae"][key], abs=1e-12)
    model = MemoryClassifier(rng=np.random.default_rng(0))
    model.load_state_dict(state("42__memory__"))
    prediction = model(Tensor(arrays["data"]["42__memory_test_x"])).data
    np.testing.assert_allclose(prediction, arrays["outputs"]["42__memory_test_logits"], atol=1e-12)


def test_learning_controls_kl_ablation_and_episode_metrics(experiment):
    _, metrics, _, arrays = experiment
    result = metrics["42"]
    assert result["vae"]["mean_latent_reconstruction_mse"] < result["baselines"]["mean_pixel_mse"]
    assert result["vae_no_kl"]["final_budget_kl"] > result["vae"]["final_budget_kl"] * 5
    assert (
        result["vae_no_kl"]["final_budget_reconstruction_nll"]
        < result["vae"]["final_budget_reconstruction_nll"]
    )
    assert result["memory"]["test_accuracy"] > 0.98
    assert result["memory"]["test_accuracy"] > result["memory"]["last_bit_baseline_accuracy"]
    for method in ("q_learning", "reinforce", "ppo"):
        returns = arrays["outputs"]["42__rl_" + method + "_returns"]
        assert returns.mean() == pytest.approx(result["rl"][method]["mean_return"])
        assert result["rl"][method]["success_rate"] > result["rl"]["random"]["success_rate"]


@pytest.mark.parametrize(
    "options", [{"seeds": []}, {"seeds": [1, 1]}, {"epochs": 0}, {"rl_updates": True}]
)
def test_demo_config_boundaries(options):
    with pytest.raises(ValueError):
        Config(**options)


def test_seeded_full_experiment_repeats_numerically(experiment, tmp_path):
    _, metrics, jsons, _ = experiment
    path, repeated = execute(Config(seeds=(42,)), tmp_path, root=tmp_path)
    assert repeated == metrics
    outputs = json.loads((path / "outputs.json").read_text())
    assert outputs == jsons["outputs"]
