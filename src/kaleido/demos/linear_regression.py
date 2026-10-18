"""P0: autodiff regression vs mean/OLS, with scale and distribution-shift controls."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kaleido.core import Tensor
from kaleido.core.gradcheck import gradcheck
from kaleido.experiments import Run, array_digest

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_train: int = 320
    n_valid: int = 96
    n_test: int = 96
    epochs: int = 300
    learning_rate: float = 0.05
    noise_std: float = 0.2
    feature_scales: tuple = (1.0, 3.0, 10.0)
    divergence_limit: float = 1e12

    def __post_init__(self):
        for name in ("seed", "n_train", "n_valid", "n_test", "epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(
                    f"{name} must be a {'non-negative' if name == 'seed' else 'positive'} integer"
                )
        for name in ("learning_rate", "noise_std", "divergence_limit"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
                or (value == 0 and name != "noise_std")
            ):
                raise ValueError(f"invalid {name}")
        scales = np.asarray(self.feature_scales)
        if scales.shape != (3,) or not np.issubdtype(scales.dtype, np.number):
            raise ValueError("feature_scales must contain three positive real numbers")
        if np.iscomplexobj(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("feature_scales must contain three positive real numbers")


def make_data(config: Config):
    """Fixed generative law; separate RNG streams for values and split membership."""
    values_seed, split_seed = np.random.SeedSequence(config.seed).spawn(2)
    rng = np.random.default_rng(values_seed)
    count = config.n_train + config.n_valid + config.n_test
    x = rng.normal(size=(count, 3)) * np.asarray(config.feature_scales)
    y = x @ np.array([[2.0], [-1.0], [0.5]]) + 0.7
    y += rng.normal(scale=config.noise_std, size=(count, 1))
    order = np.random.default_rng(split_seed).permutation(count)
    splits = np.split(order, [config.n_train, config.n_train + config.n_valid])
    return x, y, dict(zip(("train", "valid", "test"), splits, strict=True))


def standardize(train, *others):
    center, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return [(array - center) / scale for array in (train, *others)], center, scale


def mse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    return float(np.mean((prediction - target) ** 2))


def stable_learning_rate_limit(x):
    """For MSE, Hessian=2 A^T A/n; full-batch GD is stable for 0 < lr < 2/lmax."""
    design = np.column_stack([x, np.ones(len(x))])
    largest = np.linalg.eigvalsh(2 * design.T @ design / len(x))[-1]
    return float(2 / largest)


def fit(x, y, valid_x, valid_y, config: Config):
    if (
        x.ndim != 2
        or y.shape != (len(x), 1)
        or valid_x.ndim != 2
        or valid_x.shape[1] != x.shape[1]
        or valid_y.shape != (len(valid_x), 1)
    ):
        raise ValueError("fit requires rank-2 features and column targets with matching samples")
    w, b = Tensor(np.zeros((x.shape[1], 1)), True), Tensor(0.0, True)
    tx, ty = Tensor(x), Tensor(y)
    history = [
        {
            "epoch": 0,
            "train_mse": mse(x @ w.data + b.data, y),
            "valid_mse": mse(valid_x @ w.data + b.data, valid_y),
        }
    ]
    best = {
        "epoch": 0,
        "valid_mse": history[0]["valid_mse"],
        "w": w.data.copy(),
        "b": float(b.data),
    }
    for epoch in range(1, config.epochs + 1):
        loss = ((tx @ w + b - ty) ** 2).mean()
        loss.backward()
        w = Tensor(w.data - config.learning_rate * w.grad, True)
        b = Tensor(b.data - config.learning_rate * b.grad, True)
        train_mse, valid_mse = mse(x @ w.data + b.data, y), mse(valid_x @ w.data + b.data, valid_y)
        if not np.isfinite([train_mse, valid_mse]).all():
            raise FloatingPointError("training produced a non-finite loss")
        history.append({"epoch": epoch, "train_mse": train_mse, "valid_mse": valid_mse})
        if train_mse > config.divergence_limit:
            return {
                "status": "diverged",
                "history": history,
                "best": best,
                "final_w": w.data,
                "final_b": float(b.data),
            }
        if valid_mse < best["valid_mse"]:
            best = {"epoch": epoch, "valid_mse": valid_mse, "w": w.data.copy(), "b": float(b.data)}
    return {
        "status": "completed",
        "history": history,
        "best": best,
        "final_w": w.data,
        "final_b": float(b.data),
    }


def execute(config: Config, output: Path, *, root=ROOT):
    with Run(output, asdict(config), root=root) as run:
        started = time.perf_counter()
        x, y, splits = make_data(config)
        train, valid, test = (splits[name] for name in ("train", "valid", "test"))
        (train_x, valid_x, test_x), center, scale = standardize(x[train], x[valid], x[test])
        run.record(
            "data.json",
            {
                "generator": "linear-gaussian-v1",
                "sha256": array_digest(x, y, train, valid, test),
                "x": x.tolist(),
                "y": y.tolist(),
                "splits": {key: value.tolist() for key, value in splits.items()},
                "train_center": center.tolist(),
                "train_scale": scale.tolist(),
            },
        )
        sample_x, sample_y = train_x[:5], y[train][:5]
        gradient_error = gradcheck(
            lambda w, b: ((Tensor(sample_x) @ w + b - sample_y) ** 2).mean(),
            [np.zeros((3, 1)), np.array(0.0)],
        )
        trained = fit(train_x, y[train], valid_x, y[valid], config)
        if trained["status"] != "completed":
            raise RuntimeError("primary standardized training diverged; inspect learning_rate")
        ablation = fit(x[train], y[train], x[valid], y[valid], config)
        # The oracle uses training samples only; lstsq avoids explicitly inverting X^T X.
        design = np.column_stack([train_x, np.ones(len(train))])
        ols = np.linalg.lstsq(design, y[train], rcond=None)[0]
        best = trained["best"]

        def predict(features):
            return features @ best["w"] + best["b"]

        test_prediction = predict(test_x)
        ols_prediction = np.column_stack([test_x, np.ones(len(test))]) @ ols
        mean_prediction = np.full_like(y[test], y[train].mean())
        final_parameters = np.vstack([trained["final_w"], [[trained["final_b"]]]])
        metrics = {
            "train_mse": mse(predict(train_x), y[train]),
            "valid_mse": best["valid_mse"],
            "test_mse": mse(test_prediction, y[test]),
            "mean_test_mse": mse(mean_prediction, y[test]),
            "ols_test_mse": mse(ols_prediction, y[test]),
            "label_shift_plus_5_test_mse": mse(test_prediction, y[test] + 5),
            "selected_epoch": best["epoch"],
            "final_parameter_max_abs_error_vs_ols": float(np.max(np.abs(final_parameters - ols))),
            "gradcheck_max_abs_error": gradient_error,
            "stable_lr_upper_bound": {
                "standardized": stable_learning_rate_limit(train_x),
                "raw": stable_learning_rate_limit(x[train]),
            },
            "raw_scale_ablation": {
                "status": ablation["status"],
                "stopped_epoch": ablation["history"][-1]["epoch"],
                "last_train_mse": ablation["history"][-1]["train_mse"],
            },
        }
        run.record("metrics.json", metrics)
        run.record("history.json", {"standardized": trained["history"], "raw": ablation["history"]})
        raw_w = best["w"].ravel() / scale
        run.record(
            "model.json",
            {
                "standardized_weights": best["w"].ravel().tolist(),
                "standardized_bias": best["b"],
                "raw_weights": raw_w.tolist(),
                "raw_bias": float(best["b"] - center @ raw_w),
                "ols_standardized_parameters": ols.ravel().tolist(),
                "selected_epoch": best["epoch"],
                "true_raw_weights": [2.0, -1.0, 0.5],
                "true_raw_bias": 0.7,
            },
        )
        run.record(
            "predictions.json",
            [
                {
                    "sample_id": int(index),
                    "target": float(target),
                    "prediction": float(prediction),
                    "ols_prediction": float(reference),
                    "mean_prediction": float(baseline),
                }
                for index, target, prediction, reference, baseline in zip(
                    test,
                    y[test].ravel(),
                    test_prediction.ravel(),
                    ols_prediction.ravel(),
                    mean_prediction.ravel(),
                    strict=True,
                )
            ],
        )
        run.record(
            "cost.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "device": "cpu",
                "dtype": "float64",
                "primary_updates": config.epochs,
                "ablation_updates": ablation["history"][-1]["epoch"],
            },
        )
    return run.path, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p0_linear.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/p0-linear"))
    args = parser.parse_args()
    try:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    path, metrics = execute(config, args.output)
    print(json.dumps({"run": str(path.resolve()), **metrics}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
