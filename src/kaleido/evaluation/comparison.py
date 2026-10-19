"""Predeclared within-seed tests and across-fixed-seed conditional intervals."""

import numpy as np

from kaleido.evaluation.statistics import holm_adjust, mcnemar_exact, paired_question_bootstrap


def compare(results, seeds, *, resamples=4000):
    paired, intervals = {}, {}
    identities = [
        results[f"{method}-{seed}"]["identity"]
        for seed in seeds
        for method in ("sft", "grpo", "rejection_sft")
    ]
    if len({(r["case_sha256"], r["protocol_sha256"]) for r in identities}) != 1:
        raise ValueError("paired comparison requires identical case and generation protocols")
    for method in ("grpo", "rejection_sft"):
        reference = results[f"sft-{seeds[0]}"]["records"]
        suites = sorted({r["suite"] for r in reference})
        for suite in suites:
            rows = [r for r in reference if r["suite"] == suite]
            ids = [r["id"] for r in rows]
            before, after = [], []
            for seed in seeds:
                maps = [
                    {r["id"]: r for r in results[f"{name}-{seed}"]["records"]}
                    for name in ("sft", method)
                ]
                if any(set(m) != {r["id"] for r in reference} for m in maps):
                    raise ValueError("paired models must share exact evaluation IDs")
                a, b = [np.array([m[i]["correct"] for i in ids], bool) for m in maps]
                key = f"{suite}:{method}:{seed}"
                paired[key] = mcnemar_exact(a, b)
                before.append(a)
                after.append(b)
            intervals[f"{suite}:{method}"] = paired_question_bootstrap(
                before, after, [r["family"] for r in rows], resamples=resamples
            )
    corrected = holm_adjust({k: v["exact_two_sided_p"] for k, v in paired.items()})
    for key, value in corrected.items():
        paired[key]["holm_p"] = value
    return {
        "tests": paired,
        "conditional_intervals": intervals,
        "multiplicity": f"Holm over all {len(paired)} suite/method/seed tests in this panel",
        "warning": "dependent seeds/questions are not pooled into one McNemar test",
    }
