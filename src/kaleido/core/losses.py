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


def cross_entropy(logits: Tensor, target):
    """Mean multiclass NLL, shape (N,C), integer labels (N,)."""
    y = np.asarray(target)
    if len(logits.shape) != 2 or not all(logits.shape):
        raise ValueError("cross entropy requires non-empty rank-2 logits")
    if y.shape != (logits.shape[0],) or y.dtype.kind not in "iu":
        raise ValueError("targets must be an integer vector matching the batch")
    if np.any(y < 0) or np.any(y >= logits.shape[1]):
        raise ValueError("class index out of range")
    log_probability = logits - logits.logsumexp(axis=1, keepdims=True)
    return -log_probability[np.arange(len(y)), y].mean()
