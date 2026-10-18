"""K-means++/Lloyd and diagonal-covariance Gaussian-mixture EM."""

import warnings

import numpy as np

from kaleido.classical._validation import (
    FittedFeatures,
    features,
    nonnegative_float,
    positive_int,
)


def _squared_distances(x, centers):
    # Direct differences avoid cancellation for nearby points with large offsets.
    with np.errstate(over="raise", invalid="raise"):
        return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)


class KMeans(FittedFeatures):
    def __init__(self, n_clusters=3, *, n_init=10, max_iter=300, tol=1e-6, seed=0, init=None):
        for name, value in (("n_clusters", n_clusters), ("n_init", n_init), ("max_iter", max_iter)):
            positive_int(value, name)
        positive_int(seed, "seed", minimum=0)
        nonnegative_float(tol, "tol")
        if init is not None and n_init != 1:
            raise ValueError("explicit init requires n_init=1")
        self.n_clusters, self.n_init, self.max_iter = n_clusters, n_init, max_iter
        self.tol, self.seed, self.init = tol, seed, init

    def _initialize(self, x, rng):
        if self.init is not None:
            centers = features(self.init)
            if centers.shape != (self.n_clusters, x.shape[1]):
                raise ValueError("initial centers have the wrong shape")
            return centers.copy()
        centers = [x[rng.integers(len(x))].copy()]
        for _ in range(1, self.n_clusters):
            distances = _squared_distances(x, np.array(centers)).min(axis=1)
            total = distances.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError("k-means++ distances must have a positive finite sum")
            centers.append(x[rng.choice(len(x), p=distances / total)].copy())
        return np.array(centers)

    def fit(self, x):
        x = features(x)
        if len(np.unique(x, axis=0)) < self.n_clusters:
            raise ValueError("n_clusters must not exceed the number of distinct samples")
        rng, best = np.random.default_rng(self.seed), None
        for _ in range(self.n_init):
            centers = self._initialize(x, rng)
            initial = centers.copy()
            history, converged = [], False
            for _update in range(self.max_iter):
                distances = _squared_distances(x, centers)
                labels = distances.argmin(axis=1)
                counts = np.bincount(labels, minlength=self.n_clusters)
                # Empty clusters take a farthest point from a non-singleton donor.
                for empty in np.flatnonzero(counts == 0):
                    eligible = counts[labels] > 1
                    scores = np.where(eligible, distances[np.arange(len(x)), labels], -np.inf)
                    farthest = int(scores.argmax())
                    counts[labels[farthest]] -= 1
                    labels[farthest] = empty
                    counts[empty] += 1
                updated = np.array([x[labels == k].mean(axis=0) for k in range(self.n_clusters)])
                inertia = float(np.sum((x - updated[labels]) ** 2))
                history.append(inertia)
                shift = np.linalg.norm(updated - centers, axis=1).max()
                centers = updated
                if shift <= self.tol:
                    converged = True
                    break
            distances = _squared_distances(x, centers)
            labels = distances.argmin(axis=1)
            inertia = float(distances[np.arange(len(x)), labels].sum())
            candidate = (inertia, centers, labels, len(history), converged, history, initial)
            if best is None or inertia < best[0]:
                best = candidate
        (
            self.inertia_,
            self.cluster_centers_,
            self.labels_,
            self.n_iter_,
            self.converged_,
            self.inertia_history_,
            self.initial_centers_,
        ) = best
        self.n_features_in_ = x.shape[1]
        if not self.converged_:
            warnings.warn(
                "k-means reached max_iter before center tolerance", RuntimeWarning, stacklevel=2
            )
        return self

    def transform(self, x):
        return np.sqrt(_squared_distances(self._predict_x(x), self.cluster_centers_))

    def predict(self, x):
        return self.transform(x).argmin(axis=1)

    def fit_predict(self, x):
        return self.fit(x).labels_.copy()


class GaussianMixture(FittedFeatures):
    """Diagonal EM; multiple K-means starts, log-domain E-step, variance floor.

    Diagonal covariance is a modeling assumption, not a general GMM implementation.
    Components with negligible mass fail that restart rather than silently dividing by zero.
    """

    def __init__(
        self,
        n_components=3,
        *,
        n_init=3,
        max_iter=200,
        tol=1e-6,
        reg_covar=1e-6,
        seed=0,
        means_init=None,
    ):
        for name, value in (
            ("n_components", n_components),
            ("n_init", n_init),
            ("max_iter", max_iter),
        ):
            positive_int(value, name)
        positive_int(seed, "seed", minimum=0)
        nonnegative_float(tol, "tol", positive=True)
        nonnegative_float(reg_covar, "reg_covar", positive=True)
        if means_init is not None and n_init != 1:
            raise ValueError("explicit means_init requires n_init=1")
        self.n_components, self.n_init, self.max_iter = n_components, n_init, max_iter
        self.tol, self.reg_covar, self.seed = tol, reg_covar, seed
        self.means_init = means_init

    @staticmethod
    def _expectation(x, weights, means, variances):
        log_probability = -0.5 * (
            x.shape[1] * np.log(2 * np.pi)
            + np.log(variances).sum(axis=1)
            + (((x[:, None, :] - means) ** 2) / variances).sum(axis=2)
        ) + np.log(weights)
        maximum = log_probability.max(axis=1, keepdims=True)
        log_total = maximum + np.log(np.exp(log_probability - maximum).sum(axis=1, keepdims=True))
        if not np.isfinite(log_total).all():
            raise FloatingPointError("mixture likelihood is not finite")
        return log_total.ravel(), np.exp(log_probability - log_total)

    def fit(self, x):
        x = features(x, min_samples=2)
        if self.n_components > len(np.unique(x, axis=0)):
            raise ValueError("n_components must not exceed the number of distinct samples")
        rng, best, failed = np.random.default_rng(self.seed), None, 0
        for _ in range(self.n_init):
            if self.means_init is None:
                means = (
                    KMeans(self.n_components, n_init=1, seed=int(rng.integers(2**31)))
                    .fit(x)
                    .cluster_centers_.copy()
                )
            else:
                means = features(self.means_init).copy()
                if means.shape != (self.n_components, x.shape[1]):
                    raise ValueError("initial means have the wrong shape")
            weights = np.full(self.n_components, 1 / self.n_components)
            variances = np.tile(x.var(axis=0) + self.reg_covar, (self.n_components, 1))
            initial = {
                "weights": weights.copy(),
                "means": means.copy(),
                "variances": variances.copy(),
            }
            log_total, responsibility = self._expectation(x, weights, means, variances)
            history, converged, collapsed = [float(log_total.mean())], False, False
            for _update in range(self.max_iter):
                mass = responsibility.sum(axis=0)
                if np.any(mass <= np.finfo(float).eps * len(x)):
                    failed += 1
                    collapsed = True
                    break
                weights = mass / len(x)
                means = responsibility.T @ x / mass[:, None]
                variances = np.array(
                    [
                        (responsibility[:, k, None] * (x - means[k]) ** 2).sum(axis=0) / mass[k]
                        + self.reg_covar
                        for k in range(self.n_components)
                    ]
                )
                log_total, responsibility = self._expectation(x, weights, means, variances)
                history.append(float(log_total.mean()))
                if abs(history[-1] - history[-2]) <= self.tol:
                    converged = True
                    break
            if collapsed:
                continue
            candidate = (
                history[-1],
                weights,
                means,
                variances,
                len(history) - 1,
                converged,
                history,
                initial,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            raise RuntimeError(
                "all EM restarts collapsed; revise initialization or covariance floor"
            )
        (
            self.lower_bound_,
            self.weights_,
            self.means_,
            self.covariances_,
            self.n_iter_,
            self.converged_,
            self.lower_bounds_,
            self.initialization_,
        ) = best
        self.failed_restarts_, self.n_features_in_ = failed, x.shape[1]
        if not self.converged_:
            warnings.warn(
                "EM reached max_iter before likelihood tolerance", RuntimeWarning, stacklevel=2
            )
        return self

    def score_samples(self, x):
        x = self._predict_x(x)
        return self._expectation(x, self.weights_, self.means_, self.covariances_)[0]

    def predict_proba(self, x):
        x = self._predict_x(x)
        return self._expectation(x, self.weights_, self.means_, self.covariances_)[1]

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def fit_predict(self, x):
        return self.fit(x).predict(x)
