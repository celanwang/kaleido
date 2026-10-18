"""Paired inference conditional on frozen seeds; questions remain the sampling units."""

import math

import numpy as np


def mcnemar_exact(before, after):
    before = np.asarray(before)
    after = np.asarray(after)
    if (
        before.ndim != 1
        or before.shape != after.shape
        or not before.size
        or before.dtype != bool
        or after.dtype != bool
    ):
        raise ValueError("paired McNemar needs matching non-empty boolean vectors")
    wins = int(np.sum(~before & after))
    losses = int(np.sum(before & ~after))
    n = wins + losses
    p = (
        1.0
        if n == 0
        else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
    )
    return {
        "wins": wins,
        "losses": losses,
        "discordant": n,
        "delta_accuracy": float(after.mean() - before.mean()),
        "exact_two_sided_p": p,
        "unit": "paired question within one seed",
    }


def holm_adjust(pvalues):
    if not pvalues or any(not np.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("Holm adjustment requires finite probabilities")
    ordered = sorted(pvalues, key=lambda name: (pvalues[name], name))
    result = {}
    previous = 0.0
    for i, name in enumerate(ordered):
        previous = max(previous, (len(ordered) - i) * pvalues[name])
        result[name] = min(1.0, previous)
    return result


def paired_question_bootstrap(before, after, families, *, seed=1701, resamples=4000):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    families = np.asarray(families)
    if (
        before.ndim != 2
        or before.shape != after.shape
        or min(before.shape) < 1
        or families.shape != (before.shape[1],)
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or np.any((before < 0) | (before > 1))
        or np.any((after < 0) | (after > 1))
        or type(resamples) is not int
        or resamples < 100
    ):
        raise ValueError("bootstrap requires seed-by-question score matrices and family labels")
    rng = np.random.default_rng(seed)
    delta = after - before
    indices = [np.flatnonzero(families == family) for family in sorted(set(families))]
    draws = np.zeros(resamples)
    for group in indices:
        sample = rng.choice(group, size=(resamples, len(group)), replace=True)
        draws += delta[:, sample].mean(axis=0).sum(axis=1)
    draws /= before.shape[1]
    return {
        "delta_accuracy": float(delta.mean()),
        "percentile_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "resamples": resamples,
        "rng_seed": seed,
        "questions": before.shape[1],
        "fixed_seeds": before.shape[0],
        "resampling_unit": "question; retain all fixed seeds; resample within family",
        "scope": "conditional on fixed seeds/families; percentile interval, no wider guarantee",
    }


def document_nll_bootstrap(rows, *, seed=1701, resamples=4000):
    if type(resamples) is not int or resamples < 100:
        raise ValueError("at least 100 resamples required")
    if not rows or any(
        r["tokens"] < 1 or not np.isfinite(r["nll_sum"]) or r["nll_sum"] < 0 for r in rows
    ):
        raise ValueError("likelihood records need finite sums and positive token counts")
    rng = np.random.default_rng(seed)
    nll = np.array([r["nll_sum"] for r in rows])
    tokens = np.array([r["tokens"] for r in rows])
    sample = rng.integers(0, len(rows), size=(resamples, len(rows)))
    draws = nll[sample].sum(axis=1) / tokens[sample].sum(axis=1)
    mean = float(nll.sum() / tokens.sum())
    return {
        "documents": len(rows),
        "tokens": int(tokens.sum()),
        "nll": mean,
        "perplexity": math.exp(mean),
        "document_percentile_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "unit": "whole chapter, token-weighted ratio; not independent windows",
        "warning": "few related chapters; interval is descriptive conditional resampling",
    }
