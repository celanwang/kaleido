"""Single-sequence, single-head attention: a bounded bridge to the P3 decoder."""

import numpy as np

from kaleido.core.nn import Linear, Module


def scaled_dot_product_attention(query, key, value, *, mask=None):
    if (
        len(query.shape) != 2
        or len(key.shape) != 2
        or len(value.shape) != 2
        or query.shape[1] != key.shape[1]
        or key.shape[0] != value.shape[0]
        or min(*query.shape, *key.shape, *value.shape) < 1
    ):
        raise ValueError("attention requires compatible non-empty rank-2 Q/K/V")
    probability = (query @ key.T / np.sqrt(query.shape[1])).softmax(axis=1, mask=mask)
    return probability @ value, probability


class SelfAttention(Module):
    def __init__(self, width, *, rng, causal=True):
        self.query = Linear(width, width, rng=rng, bias=False)
        self.key = Linear(width, width, rng=rng, bias=False)
        self.value = Linear(width, width, rng=rng, bias=False)
        self.output = Linear(width, width, rng=rng, bias=False)
        self.causal = bool(causal)

    def forward(self, x):
        mask = np.tril(np.ones((x.shape[0], x.shape[0]), bool)) if self.causal else None
        values, _ = scaled_dot_product_attention(
            self.query(x), self.key(x), self.value(x), mask=mask
        )
        return self.output(values)
