"""P1 offline dataset benchmarks, validation selection and controlled counterexamples."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kaleido.classical import (
    PCA,
    DecisionTreeClassifier,
    GaussianMixture,
    KMeans,
    LinearRegression,
    LogisticRegression,
    StandardScaler,
)
from kaleido.classical._validation import nonnegative_float, positive_int
from kaleido.experiments import Run, array_digest

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    seeds: tuple = (42, 7, 2026)
    l2_candidates: tuple = (0.001, 0.01, 0.1)
    tree_depths: tuple = (1, 2, 3, 5, 8)
    tree_min_samples_leaf: int = 2
    pca_components: int = 2
    n_clusters: int = 3
    mixture_components: tuple = (1, 2, 3, 4)
    n_init: int = 5
    max_iter: int = 1000
    tol: float = 1e-10
    failure_seed: int = 123

    def __post_init__(self):
        for name in ("seeds", "l2_candidates", "tree_depths", "mixture_components"):
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list))
                or not values
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{name} must be a non-empty sequence of unique values")
            for value in values:
                if name == "l2_candidates":
                    nonnegative_float(value, name, positive=True)
                else:
                    positive_int(value, name, minimum=0 if name == "seeds" else 1)
        for name in ("tree_min_samples_leaf", "pca_components", "n_clusters", "n_init", "max_iter"):
            positive_int(getattr(self, name), name)
        positive_int(self.failure_seed, "failure_seed", minimum=0)
        nonnegative_float(self.tol, "tol", positive=True)
        if max(self.tree_depths) > 64:
            raise ValueError("tree depths must not exceed 64")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _split(y, seed, *, classification):
    from sklearn.model_selection import train_test_split

    train, held = train_test_split(
        np.arange(len(y)), test_size=0.4, random_state=seed, stratify=y if classification else None
    )
    valid, test = train_test_split(
        held, test_size=0.5, random_state=seed + 1, stratify=y[held] if classification else None
    )
    return {"train": train, "valid": valid, "test": test}


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _accuracy(a, b):
    return float(np.mean(a == b))


def _nll(logits, y):
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def _parameters(model, *names):
    return {name: getattr(model, name) for name in names}


def _supervised(config, datasets, splits):
    from sklearn.linear_model import LinearRegression as ReferenceLinear
    from sklearn.linear_model import LogisticRegression as ReferenceLogistic
    from sklearn.preprocessing import StandardScaler as ReferenceScaler
    from sklearn.tree import DecisionTreeClassifier as ReferenceTree

    result = {
        key: {} for key in ("metrics", "selection", "models", "predictions", "histories", "checks")
    }
    for name in ("diabetes", "breast_cancer"):
        x, y = datasets[name].data, datasets[name].target
        tr, va, te = (splits[name][key] for key in ("train", "valid", "test"))
        scaler = StandardScaler().fit(x[tr])
        tx, vx, ex = (scaler.transform(x[ids]) for ids in (tr, va, te))
        reference_scaler = ReferenceScaler().fit(x[tr])
        result["checks"][name + "_scaler_max_abs_error"] = float(
            np.max(np.abs(ex - reference_scaler.transform(x[te])))
        )
        if name == "diabetes":
            model, reference = LinearRegression().fit(tx, y[tr]), ReferenceLinear().fit(tx, y[tr])
            prediction, expected = model.predict(ex), reference.predict(ex)
            result["metrics"]["linear"] = {
                "test_mse": _mse(prediction, y[te]),
                "reference_test_mse": _mse(expected, y[te]),
                "mean_baseline_test_mse": _mse(np.full(len(te), y[tr].mean()), y[te]),
                "test_mae": float(np.mean(np.abs(prediction - y[te]))),
            }
            result["checks"]["linear_prediction_max_abs_error"] = float(
                np.max(np.abs(prediction - expected))
            )
        else:
            candidates, fitted = [], []
            for l2 in config.l2_candidates:
                candidate = LogisticRegression(l2=l2, max_iter=config.max_iter, tol=1e-9).fit(
                    tx, y[tr]
                )
                if not candidate.converged_:
                    raise RuntimeError("logistic candidate failed to converge")
                candidates.append(
                    {"l2": l2, "valid_nll": _nll(candidate.decision_function(vx), y[va])}
                )
                fitted.append(candidate)
            chosen = min(range(len(candidates)), key=lambda i: candidates[i]["valid_nll"])
            model = fitted[chosen]
            reference = ReferenceLogistic(C=1 / (len(tr) * model.l2), tol=1e-10, max_iter=3000).fit(
                tx, y[tr]
            )
            prediction, expected = model.predict(ex), reference.predict(ex)
            result["selection"]["logistic"] = {
                "candidates": candidates,
                "selected_l2": model.l2,
                "reference_C": 1 / (len(tr) * model.l2),
            }
            result["histories"]["logistic_loss"] = model.loss_history_
            result["metrics"]["logistic"] = {
                "test_accuracy": _accuracy(prediction, y[te]),
                "reference_test_accuracy": _accuracy(expected, y[te]),
                "test_nll": _nll(model.decision_function(ex), y[te]),
                "reference_test_nll": _nll(reference.decision_function(ex), y[te]),
                "majority_baseline_test_accuracy": _accuracy(
                    np.full(len(te), np.bincount(y[tr]).argmax()), y[te]
                ),
            }
            result["checks"]["logistic_probability_max_abs_error"] = float(
                np.max(np.abs(model.predict_proba(ex) - reference.predict_proba(ex)))
            )
            result["checks"]["logistic_gradient_max_abs"] = model.gradient_max_abs_
        result["models"][name] = {
            "estimator": _parameters(model, "coef_", "intercept_"),
            "scaler": _parameters(scaler, "mean_", "scale_"),
        }
        result["predictions"][name] = {
            "sample_ids": te,
            "target": y[te],
            "prediction": prediction,
            "reference_prediction": expected,
        }
        if name == "breast_cancer":
            result["predictions"][name].update(
                logits=model.decision_function(ex),
                probability=model.predict_proba(ex),
                reference_probability=reference.predict_proba(ex),
            )
    x, y = datasets["iris"].data, datasets["iris"].target
    tr, va, te = (splits["iris"][key] for key in ("train", "valid", "test"))
    candidates, fitted = [], []
    for depth in config.tree_depths:
        model = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=config.tree_min_samples_leaf
        ).fit(x[tr], y[tr])
        candidates.append(
            {
                "max_depth": depth,
                "valid_accuracy": _accuracy(model.predict(x[va]), y[va]),
                "train_accuracy": _accuracy(model.predict(x[tr]), y[tr]),
            }
        )
        fitted.append(model)
    chosen = min(
        range(len(candidates)),
        key=lambda i: (-candidates[i]["valid_accuracy"], candidates[i]["max_depth"]),
    )
    tree = fitted[chosen]
    reference = ReferenceTree(
        max_depth=tree.max_depth, min_samples_leaf=config.tree_min_samples_leaf, random_state=0
    ).fit(x[tr], y[tr])
    prediction, expected = tree.predict(x[te]), reference.predict(x[te])
    result["metrics"]["tree"] = {
        "test_accuracy": _accuracy(prediction, y[te]),
        "reference_test_accuracy": _accuracy(expected, y[te]),
        "prediction_agreement": _accuracy(prediction, expected),
    }
    result["selection"]["tree"] = {"candidates": candidates, "selected_depth": tree.max_depth}
    result["models"]["tree"] = {
        "classes": tree.classes_,
        "depth": tree.depth_,
        "leaves": tree.n_leaves_,
        "feature_importances": tree.feature_importances_,
        "nodes": _tree_nodes(tree.root_),
    }
    result["predictions"]["tree"] = {
        "sample_ids": te,
        "target": y[te],
        "prediction": prediction,
        "reference_prediction": expected,
        "probability": tree.predict_proba(x[te]),
    }
    return result


def _tree_nodes(node):
    result = {"samples": node.samples, "impurity": node.impurity, "probability": node.probabilities}
    if node.feature >= 0:
        result.update(
            feature=node.feature,
            threshold=node.threshold,
            left=_tree_nodes(node.left),
            right=_tree_nodes(node.right),
        )
    return result


def _unsupervised(config, dataset, splits, seed):
    from sklearn.cluster import KMeans as ReferenceKMeans
    from sklearn.decomposition import PCA as ReferencePCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture as ReferenceMixture

    result = {
        key: {} for key in ("metrics", "selection", "models", "predictions", "histories", "checks")
    }
    x, y = dataset.data, dataset.target
    tr, va, te = (splits[key] for key in ("train", "valid", "test"))
    scaler = StandardScaler().fit(x[tr])
    tx, vx, ex = (scaler.transform(x[ids]) for ids in (tr, va, te))
    pca = PCA(config.pca_components).fit(tx)
    reference_pca = ReferencePCA(config.pca_components, svd_solver="full").fit(tx)
    reconstructed = pca.inverse_transform(pca.transform(ex))
    expected = reference_pca.inverse_transform(reference_pca.transform(ex))
    result["metrics"]["pca"] = {
        "test_reconstruction_mse": _mse(reconstructed, ex),
        "reference_test_reconstruction_mse": _mse(expected, ex),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    result["checks"]["pca_projection_max_abs_error"] = float(
        np.max(
            np.abs(
                pca.components_.T @ pca.components_
                - reference_pca.components_.T @ reference_pca.components_
            )
        )
    )
    km = KMeans(
        config.n_clusters, n_init=config.n_init, seed=seed, max_iter=config.max_iter, tol=config.tol
    ).fit(tx)
    if not km.converged_:
        raise RuntimeError("k-means failed to converge")
    reference_km = ReferenceKMeans(
        config.n_clusters, init=km.initial_centers_, n_init=1, max_iter=config.max_iter, tol=1e-10
    ).fit(tx)
    distances, expected_distances = (
        km.transform(ex).min(axis=1) ** 2,
        reference_km.transform(ex).min(axis=1) ** 2,
    )
    result["metrics"]["kmeans"] = {
        "test_mean_squared_distance": float(distances.mean()),
        "reference_test_mean_squared_distance": float(expected_distances.mean()),
        "test_adjusted_rand": float(adjusted_rand_score(y[te], km.predict(ex))),
    }
    result["checks"]["kmeans_distance_max_abs_error"] = float(
        np.max(np.abs(distances - expected_distances))
    )
    candidates, fitted = [], []
    for count in config.mixture_components:
        model = GaussianMixture(
            count, n_init=config.n_init, max_iter=config.max_iter, seed=seed, tol=config.tol
        ).fit(tx)
        if not model.converged_:
            raise RuntimeError("EM candidate failed to converge")
        candidates.append(
            {
                "components": count,
                "valid_mean_log_likelihood": float(model.score_samples(vx).mean()),
            }
        )
        fitted.append(model)
    chosen = max(range(len(candidates)), key=lambda i: candidates[i]["valid_mean_log_likelihood"])
    mixture = fitted[chosen]
    initial = mixture.initialization_
    reference_mixture = ReferenceMixture(
        mixture.n_components,
        covariance_type="diag",
        n_init=1,
        tol=config.tol,
        max_iter=config.max_iter,
        reg_covar=mixture.reg_covar,
        weights_init=initial["weights"],
        means_init=initial["means"],
        precisions_init=1 / initial["variances"],
        random_state=seed,
    ).fit(tx)
    if not reference_mixture.converged_:
        raise RuntimeError("reference EM failed to converge")
    likelihood, expected_likelihood = mixture.score_samples(ex), reference_mixture.score_samples(ex)
    result["metrics"]["em"] = {
        "test_mean_log_likelihood": float(likelihood.mean()),
        "reference_test_mean_log_likelihood": float(expected_likelihood.mean()),
        "test_adjusted_rand": float(adjusted_rand_score(y[te], mixture.predict(ex))),
    }
    result["checks"]["em_likelihood_max_abs_error"] = float(
        np.max(np.abs(likelihood - expected_likelihood))
    )
    result["selection"]["em"] = {
        "candidates": candidates,
        "selected_components": mixture.n_components,
    }
    result["models"].update(
        iris_scaler=_parameters(scaler, "mean_", "scale_"),
        pca=_parameters(
            pca, "mean_", "components_", "explained_variance_", "explained_variance_ratio_"
        ),
        kmeans=_parameters(km, "cluster_centers_", "initial_centers_", "n_iter_"),
        em=_parameters(
            mixture,
            "weights_",
            "means_",
            "covariances_",
            "initialization_",
            "n_iter_",
            "failed_restarts_",
        ),
        em_reference=_parameters(
            reference_mixture, "weights_", "means_", "covariances_", "n_iter_", "converged_"
        ),
    )
    result["predictions"]["unsupervised"] = {
        "sample_ids": te,
        "target_for_reporting_only": y[te],
        "pca_reconstruction": reconstructed,
        "pca_reference_reconstruction": expected,
        "kmeans_label": km.predict(ex),
        "kmeans_squared_distance": distances,
        "kmeans_reference_squared_distance": expected_distances,
        "em_label": mixture.predict(ex),
        "em_probability": mixture.predict_proba(ex),
        "em_log_likelihood": likelihood,
        "em_reference_log_likelihood": expected_likelihood,
    }
    result["histories"] = {
        "kmeans_inertia": km.inertia_history_,
        "em_log_likelihood": mixture.lower_bounds_,
    }
    return result


def _failure_cases(seed):
    from sklearn.mixture import GaussianMixture as ReferenceMixture

    xor_x, xor_y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])
    linear = LogisticRegression().fit(xor_x, xor_y)
    tree = DecisionTreeClassifier(max_depth=2).fit(xor_x, xor_y)
    gated = DecisionTreeClassifier(max_depth=2, min_impurity_decrease=0.01).fit(xor_x, xor_y)
    rng = np.random.default_rng(seed)
    train_y, test_y = rng.integers(0, 2, 600), rng.integers(0, 2, 300)
    train_x = np.column_stack(
        [
            rng.normal(0, 20, len(train_y)),
            (train_y * 2 - 1) * 0.1 + rng.normal(0, 0.01, len(train_y)),
        ]
    )
    test_x = np.column_stack(
        [rng.normal(0, 20, len(test_y)), (test_y * 2 - 1) * 0.1 + rng.normal(0, 0.01, len(test_y))]
    )
    pca = PCA(1).fit(train_x)
    scale = StandardScaler().fit(train_x)
    full = LogisticRegression().fit(scale.transform(train_x), train_y)
    reduced_scale = StandardScaler().fit(pca.transform(train_x))
    reduced = LogisticRegression().fit(reduced_scale.transform(pca.transform(train_x)), train_y)
    full_prediction = full.predict(scale.transform(test_x))
    reduced_prediction = reduced.predict(reduced_scale.transform(pca.transform(test_x)))
    gaussian = rng.multivariate_normal([0, 0], [[4, 3.6], [3.6, 4]], size=600)
    train, test = gaussian[:400], gaussian[400:]
    diagonal = GaussianMixture(1, n_init=1, seed=seed).fit(train)
    correlated = ReferenceMixture(1, covariance_type="full", random_state=seed).fit(train)
    return {
        "xor": {
            "x": xor_x,
            "target": xor_y,
            "logistic_prediction": linear.predict(xor_x),
            "tree_prediction": tree.predict(xor_x),
            "positive_gain_gate_prediction": gated.predict(xor_x),
        },
        "pca_discards_signal": {
            "data_sha256": array_digest(train_x, train_y, test_x, test_y),
            "train_x": train_x,
            "train_y": train_y,
            "test_x": test_x,
            "test_y": test_y,
            "full_accuracy": _accuracy(full_prediction, test_y),
            "one_component_accuracy": _accuracy(reduced_prediction, test_y),
            "retained_variance_ratio": pca.explained_variance_ratio_[0],
            "full_prediction": full_prediction,
            "pca_prediction": reduced_prediction,
        },
        "diagonal_misses_correlation": {
            "data_sha256": array_digest(train, test),
            "train_x": train,
            "test_x": test,
            "diagonal_mean_test_log_likelihood": float(diagonal.score_samples(test).mean()),
            "full_mean_test_log_likelihood": float(correlated.score_samples(test).mean()),
            "diagonal_test_log_likelihood": diagonal.score_samples(test),
            "full_test_log_likelihood": correlated.score_samples(test),
        },
    }


def execute(config: Config, output: Path, *, root=ROOT):
    try:
        from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
    except ImportError as error:
        raise RuntimeError(
            "install offline reference dependencies: uv sync --locked --extra reference"
        ) from error
    with Run(output, asdict(config), root=root) as run:
        started = time.perf_counter()
        datasets = {
            "diabetes": load_diabetes(scaled=False),
            "breast_cancer": load_breast_cancer(),
            "iris": load_iris(),
        }
        data = {
            name: {
                "x": ds.data,
                "y": ds.target,
                "feature_names": list(ds.feature_names),
                "sha256": array_digest(ds.data, ds.target),
                "provider": "scikit-learn bundled dataset",
                "source": "https://scikit-learn.org/stable/datasets/toy_dataset.html",
            }
            for name, ds in datasets.items()
        }
        artifacts = {
            key: {}
            for key in (
                "metrics",
                "selection",
                "models",
                "predictions",
                "histories",
                "checks",
                "splits",
            )
        }
        for seed in config.seeds:
            splits = {
                name: _split(ds.target, seed, classification=name != "diabetes")
                for name, ds in datasets.items()
            }
            supervised = _supervised(config, datasets, splits)
            unsupervised = _unsupervised(config, datasets["iris"], splits["iris"], seed)
            for key in supervised:
                artifacts[key][str(seed)] = {**supervised[key], **unsupervised[key]}
            artifacts["splits"][str(seed)] = splits
        failures = _failure_cases(config.failure_seed)
        summary = {
            "n_splits": len(config.seeds),
            "seeds": config.seeds,
            "dispersion": (
                "sample standard deviation across overlapping seeded splits; "
                "not a confidence interval"
            ),
            "methods": {},
        }
        first = artifacts["metrics"][str(config.seeds[0])]
        for method, values in first.items():
            summary["methods"][method] = {}
            for metric in values:
                samples = [artifacts["metrics"][str(seed)][method][metric] for seed in config.seeds]
                summary["methods"][method][metric] = {
                    "mean": float(np.mean(samples)),
                    "split_std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
                }
        run.record(
            "data.json",
            _jsonable(
                {
                    "datasets": data,
                    "splits": artifacts.pop("splits"),
                    "protocol": "60/20/20; stratified classification; training-only scaling",
                }
            ),
        )
        for key, value in artifacts.items():
            run.record(key + ".json", _jsonable(value))
        run.record("failure_cases.json", _jsonable(failures))
        run.record("summary.json", _jsonable(summary))
        run.record(
            "cost.json",
            {"wall_seconds": time.perf_counter() - started, "device": "cpu", "dtype": "float64"},
        )
        thresholds = {
            "diabetes_scaler_max_abs_error": 1e-10,
            "breast_cancer_scaler_max_abs_error": 1e-10,
            "linear_prediction_max_abs_error": 1e-8,
            "logistic_probability_max_abs_error": 2e-5,
            "logistic_gradient_max_abs": 1e-8,
            "pca_projection_max_abs_error": 1e-10,
            "kmeans_distance_max_abs_error": 1e-8,
            "em_likelihood_max_abs_error": 1e-4,
        }
        for seed, checks in artifacts["checks"].items():
            for name, value in checks.items():
                if value > thresholds[name]:
                    raise RuntimeError(f"reference check failed for split {seed}: {name}={value}")
    return run.path, _jsonable(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p1_classical.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/p1-classical"))
    args = parser.parse_args()
    try:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    path, summary = execute(config, args.output)
    print(json.dumps({"run": str(path.resolve()), **summary}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
