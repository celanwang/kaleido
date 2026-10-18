"""Small stable numerical primitives shared by the learning implementations."""

import numpy as np


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    exp_negative_abs = np.exp(-np.abs(value))
    return np.where(
        value >= 0, 1 / (1 + exp_negative_abs), exp_negative_abs / (1 + exp_negative_abs)
    )
