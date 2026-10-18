import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from kaleido.demos.classical_ml import Config, execute  # noqa: E402
from kaleido.experiments import array_digest  # noqa: E402


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    output = tmp_path_factory.mktemp("p1")
    path, summary = execute(Config(), output, root=output)
    artifacts = {file.stem: json.loads(file.read_text()) for file in path.glob("*.json")}
    return artifacts, summary


def test_complete_benchmark_artifacts_and_train_only_preprocessing(benchmark):
    artifacts, _ = benchmark
    assert artifacts["manifest"]["status"] == "completed"
    for name, dataset in artifacts["data"]["datasets"].items():
        x, y = np.array(dataset["x"]), np.array(dataset["y"])
        assert array_digest(x, y) == dataset["sha256"]
        for seed, all_splits in artifacts["data"]["splits"].items():
            splits = all_splits[name]
            assert sorted(sum(splits.values(), [])) == list(range(len(x)))
            model_name = "iris_scaler" if name == "iris" else name
            stored = artifacts["models"][seed][model_name]
            scaler = stored if name == "iris" else stored["scaler"]
            np.testing.assert_allclose(scaler["mean_"], x[splits["train"]].mean(axis=0))
            np.testing.assert_allclose(scaler["scale_"], x[splits["train"]].std(axis=0))


def test_saved_sample_outputs_reconstruct_all_test_metrics(benchmark):
    artifacts, summary = benchmark
    for seed, metrics in artifacts["metrics"].items():
        predictions = artifacts["predictions"][seed]
        for name, method in (
            ("diabetes", "linear"),
            ("breast_cancer", "logistic"),
            ("tree", "tree"),
        ):
            rows = predictions[name]
            y, prediction = np.array(rows["target"]), np.array(rows["prediction"])
            value = (
                np.mean((y - prediction) ** 2) if method == "linear" else np.mean(y == prediction)
            )
            key = "test_mse" if method == "linear" else "test_accuracy"
            assert value == pytest.approx(metrics[method][key])
        iris = predictions["unsupervised"]
        assert np.mean(iris["kmeans_squared_distance"]) == pytest.approx(
            metrics["kmeans"]["test_mean_squared_distance"]
        )
        assert np.mean(iris["em_log_likelihood"]) == pytest.approx(
            metrics["em"]["test_mean_log_likelihood"]
        )
        np.testing.assert_allclose(np.array(iris["em_probability"]).sum(axis=1), 1)
        scaler = artifacts["models"][seed]["iris_scaler"]
        raw = np.array(artifacts["data"]["datasets"]["iris"]["x"])[iris["sample_ids"]]
        scaled = (raw - scaler["mean_"]) / scaler["scale_"]
        assert np.mean((np.array(iris["pca_reconstruction"]) - scaled) ** 2) == pytest.approx(
            metrics["pca"]["test_reconstruction_mse"]
        )
    for method, values in summary["methods"].items():
        for metric, aggregation in values.items():
            numbers = [metrics[method][metric] for metrics in artifacts["metrics"].values()]
            assert aggregation["mean"] == pytest.approx(np.mean(numbers))
            assert aggregation["split_std"] == pytest.approx(np.std(numbers, ddof=1))


def test_selection_uses_validation_scores_and_reference_agrees(benchmark):
    artifacts, _ = benchmark
    for seed, methods in artifacts["selection"].items():
        logistic = methods["logistic"]
        assert (
            logistic["selected_l2"]
            == min(logistic["candidates"], key=lambda row: row["valid_nll"])["l2"]
        )
        tree = methods["tree"]
        assert (
            tree["selected_depth"]
            == min(tree["candidates"], key=lambda row: (-row["valid_accuracy"], row["max_depth"]))[
                "max_depth"
            ]
        )
        em = methods["em"]
        assert (
            em["selected_components"]
            == max(em["candidates"], key=lambda row: row["valid_mean_log_likelihood"])["components"]
        )
        assert artifacts["checks"][seed]["logistic_probability_max_abs_error"] < 2e-5
        assert artifacts["checks"][seed]["em_likelihood_max_abs_error"] < 1e-4
        assert (
            artifacts["metrics"][seed]["linear"]["test_mse"]
            < artifacts["metrics"][seed]["linear"]["mean_baseline_test_mse"]
        )


def test_failure_cases_show_representation_and_covariance_limits(benchmark):
    artifacts, _ = benchmark
    cases = artifacts["failure_cases"]
    xor = cases["xor"]
    assert xor["tree_prediction"] == xor["target"]
    assert xor["logistic_prediction"] != xor["target"]
    assert cases["pca_discards_signal"]["full_accuracy"] > 0.95
    assert 0.4 < cases["pca_discards_signal"]["one_component_accuracy"] < 0.6
    assert cases["pca_discards_signal"]["retained_variance_ratio"] > 0.99
    diagonal = cases["diagonal_misses_correlation"]
    assert (
        diagonal["full_mean_test_log_likelihood"]
        > diagonal["diagonal_mean_test_log_likelihood"] + 0.3
    )


def test_rerun_matches_seeded_metrics(benchmark, tmp_path):
    artifacts, _ = benchmark
    path, _ = execute(Config(seeds=(42,)), tmp_path, root=tmp_path)
    repeated = json.loads((path / "metrics.json").read_text())
    assert repeated["42"] == artifacts["metrics"]["42"]


@pytest.mark.parametrize(
    "options",
    [
        {"seeds": []},
        {"seeds": [1, 1]},
        {"tree_depths": [65]},
        {"l2_candidates": [0]},
        {"tol": float("nan")},
        {"pca_components": True},
        {"mixture_components": [-1]},
    ],
)
def test_config_boundaries(options):
    with pytest.raises(ValueError):
        Config(**options)
