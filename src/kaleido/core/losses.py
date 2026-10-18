"""Losses operate on logits so probability rounding cannot produce log(0)."""

import numpy as np

from kaleido.core.tensor import Tensor


def binary_cross_entropy_with_logits(logits: Tensor, target):
    target = Tensor(target)
    if target.shape != logits.shape or target.data.size == 0:
        raise ValueError("targets must have the same non-empty shape as logits")
    if np.any((target.data < 0) | (target.data > 1)):
        raise ValueError("binary targets must lie in [0, 1]")
    return (logits.softplus() - logits * target).mean()
