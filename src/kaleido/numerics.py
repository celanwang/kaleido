"""Small stable numerical primitives shared by the learning implementations."""

import numpy as np


def nonnegative_float(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not np.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    exp_negative_abs = np.exp(-np.abs(value))
    return np.where(
        value >= 0, 1 / (1 + exp_negative_abs), exp_negative_abs / (1 + exp_negative_abs)
    )
