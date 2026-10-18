import json

import numpy as np
import pytest

from kaleido.demos.linear_regression import Config, execute, fit, make_data, standardize


def test_splits_are_disjoint_exhaustive_and_frozen_by_seed():
    config = Config()
    x, y, splits = make_data(config)
    again_x, again_y, again_splits = make_data(config)
    assert len(splits["train"]) == 320
    assert len(splits["valid"]) == 96
    assert len(splits["test"]) == 96
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(x)))
    np.testing.assert_array_equal(x, again_x)
    np.testing.assert_array_equal(y, again_y)
    for name in splits:
        np.testing.assert_array_equal(splits[name], again_splits[name])


def test_standardization_fits_only_training_data():
    train = np.array([[1.0, 2], [3, 2]])
    test = np.array([[101.0, 20]])
    (scaled_train, scaled_test), center, scale = standardize(train, test)
    np.testing.assert_array_equal(center, [2, 2])
    np.testing.assert_array_equal(scale, [1, 1])
    np.testing.assert_array_equal(scaled_train, [[-1, 0], [1, 0]])
    np.testing.assert_array_equal(scaled_test, [[99, 18]])


def test_complete_reproducible_training_vs_baselines_and_failure_controls(tmp_path):
    first, metrics = execute(Config(), tmp_path, root=tmp_path)
    second, repeated = execute(Config(), tmp_path, root=tmp_path)
    assert metrics == repeated
    assert first != second
    assert metrics["test_mse"] < metrics["mean_test_mse"] * 0.01
    assert abs(metrics["test_mse"] - metrics["ols_test_mse"]) < 0.01
    assert metrics["final_parameter_max_abs_error_vs_ols"] < 1e-8
    assert metrics["gradcheck_max_abs_error"] < 1e-6
    assert metrics["label_shift_plus_5_test_mse"] > metrics["test_mse"] * 100
    assert metrics["raw_scale_ablation"]["status"] == "diverged"
    assert metrics["stable_lr_upper_bound"]["raw"] < 0.05
    assert metrics["stable_lr_upper_bound"]["standardized"] > 0.05
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    history = json.loads((first / "history.json").read_text())["standardized"]
    expected_epoch = min(history, key=lambda row: row["valid_mse"])["epoch"]
    assert metrics["selected_epoch"] == expected_epoch
    predictions = json.loads((first / "predictions.json").read_text())
    computed = np.mean([(row["target"] - row["prediction"]) ** 2 for row in predictions])
    assert computed == pytest.approx(metrics["test_mse"])
    data = json.loads((first / "data.json").read_text())
    assert [row["sample_id"] for row in predictions] == data["splits"]["test"]
    model = json.loads((first / "model.json").read_text())
    x = np.array(data["x"])[data["splits"]["test"]]
    recovered = x @ np.array(model["raw_weights"]) + model["raw_bias"]
    np.testing.assert_allclose(recovered, [row["prediction"] for row in predictions])


@pytest.mark.parametrize("seed", [0, 7, 2026])
def test_training_converges_across_data_seeds(seed):
    config = Config(seed=seed)
    x, y, splits = make_data(config)
    train, valid = splits["train"], splits["valid"]
    (tx, vx), _, _ = standardize(x[train], x[valid])
    result = fit(tx, y[train], vx, y[valid], config)
    assert result["status"] == "completed"
    design = np.column_stack([tx, np.ones(len(tx))])
    ols = np.linalg.lstsq(design, y[train], rcond=None)[0]
    np.testing.assert_allclose(result["final_w"], ols[:-1], atol=1e-8)
    assert result["final_b"] == pytest.approx(ols[-1, 0], abs=1e-8)


@pytest.mark.parametrize(
    "options",
    [
        {"epochs": 0},
        {"epochs": True},
        {"seed": -1},
        {"n_test": 1.5},
        {"learning_rate": float("nan")},
        {"noise_std": -1},
        {"feature_scales": [1, 2]},
        {"feature_scales": [1, 2, 0]},
        {"feature_scales": [1, 2, "3"]},
    ],
)
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError):
        Config(**options)


def test_primary_divergence_records_failure(tmp_path):
    with pytest.raises(RuntimeError, match="diverged"):
        execute(Config(learning_rate=10), tmp_path, root=tmp_path)
    [manifest_path] = tmp_path.glob("*/manifest.json")
    assert json.loads(manifest_path.read_text())["status"] == "failed"


def test_column_target_shape_required():
    x = np.ones((5, 3))
    with pytest.raises(ValueError, match="column"):
        fit(x, np.ones(5), x, np.ones(5), Config())
