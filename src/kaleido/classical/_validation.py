"""Shared boundaries, not a generic trainer or sklearn compatibility layer."""

import numpy as np


def features(x, *, min_samples=1):
    original = np.asarray(x)
    if np.iscomplexobj(original) or not np.issubdtype(original.dtype, np.number):
        raise ValueError("features must be real numeric values")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < min_samples or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("features must be a finite non-empty rank-2 array")
    return x


def target(y, n_samples, *, numeric=False):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n_samples or y.dtype.kind not in "biufUS":
        raise ValueError("target must be a matching one-dimensional numeric or string array")
    if y.dtype.kind in "biuf" and not np.isfinite(y).all():
        raise ValueError("targets must be finite")
    if numeric:
        if y.dtype.kind not in "biuf":
            raise ValueError("regression targets must be numeric")
        return y.astype(np.float64)
    return y


def positive_int(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def nonnegative_float(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not np.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")


class FittedFeatures:
    def _predict_x(self, x):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("estimator must be fitted first")
        x = features(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("feature count differs from fitted data")
        return x
