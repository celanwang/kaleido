"""Generated integer tasks with structural splits and a non-executing verifier."""

import hashlib
import json
import re
from collections import Counter

import numpy as np

FAMILIES = {
    "train": ("sum_scale", "product_shift", "scale_sum", "difference_scale"),
    "valid": ("sum_product", "product_difference"),
    "test": ("scaled_sum_shift", "sum_products"),
}
SYSTEM = (
    "Return exactly one line: <answer>N</answer>, where N is the requested integer. "
    "Do not include explanation."
)


def expression(family, a, b, c, d, m):
    def add(x, y):
        return ("add", x, y)

    def sub(x, y):
        return ("sub", x, y)

    def mul(x, y):
        return ("mul", x, y)

    trees = {
        "sum_scale": mul(add(a, b), c),
        "product_shift": add(mul(a, b), c),
        "scale_sum": mul(a, add(b, c)),
        "difference_scale": mul(sub(a, b), c),
        "sum_product": mul(add(a, b), add(c, d)),
        "product_difference": sub(mul(a, b), mul(c, d)),
        "scaled_sum_shift": add(mul(add(a, b), c), d),
        "sum_products": add(mul(a, b), mul(c, d)),
    }
    if family not in trees:
        raise ValueError("unknown task family")
    return ("mod", trees[family], m)


def evaluate_ast(node):
    if type(node) is int:
        return node
    operation, left, right = node
    a, b = evaluate_ast(left), evaluate_ast(right)
    if operation == "add":
        return a + b
    if operation == "sub":
        return a - b
    if operation == "mul":
        return a * b
    if operation == "mod" and b > 0:
        return a % b
    raise ValueError("invalid arithmetic node")


def render(node):
    if type(node) is int:
        return str(node)
    op, a, b = node
    symbols = {"add": "+", "sub": "-", "mul": "*", "mod": "%"}
    return f"({render(a)} {symbols[op]} {render(b)})"


def independent_oracle(row):
    """Expanded polynomial oracle does not call the training AST interpreter."""
    a, b, c, d, m = (row["operands"][k] for k in ("a", "b", "c", "d", "m"))
    polynomial = {
        "sum_scale": a * c + b * c,
        "product_shift": a * b + c,
        "scale_sum": a * b + a * c,
        "difference_scale": a * c - b * c,
        "sum_product": a * c + a * d + b * c + b * d,
        "product_difference": a * b - c * d,
        "scaled_sum_shift": a * c + b * c + d,
        "sum_products": a * b + c * d,
    }[row["family"]]
    return int(polynomial - m * (polynomial // m))


def make_tasks(*, seed=731, train_per_family=32, heldout_per_family=16, operand_max=30):
    if (
        type(seed) is not int
        or seed < 0
        or any(
            type(v) is not int or v < 1 for v in (train_per_family, heldout_per_family, operand_max)
        )
    ):
        raise ValueError("task counts and operand range must be positive integers")
    rng = np.random.default_rng(seed)
    tasks = []
    seen = set()
    for split, families in FAMILIES.items():
        count = train_per_family if split == "train" else heldout_per_family
        for family in families:
            attempts = 0
            accepted = 0
            while accepted < count:
                attempts += 1
                if attempts > count * 10000:
                    raise ValueError("not enough unique arithmetic tasks")
                a, b, c, d = map(int, rng.integers(1, operand_max + 1, size=4))
                m = int(rng.integers(8, 18))
                tree = expression(family, a, b, c, d, m)
                formula = render(tree)
                answer = evaluate_ast(tree)
                # Balance each family's labels before freezing any evaluation outputs.
                if answer != accepted % 8:
                    continue
                if formula in seen:
                    continue
                seen.add(formula)
                row = {
                    "id": hashlib.sha256(formula.encode()).hexdigest()[:20],
                    "split": split,
                    "family": family,
                    "expression": formula,
                    "operands": dict(zip(("a", "b", "c", "d", "m"), (a, b, c, d, m), strict=True)),
                    "prompt": (
                        f"Compute {formula}. The symbol % means the least non-negative remainder."
                    ),
                    "answer": answer,
                    "completion": f"<answer>{answer}</answer>",
                }
                if independent_oracle(row) != answer:
                    raise AssertionError("independent arithmetic oracles disagree")
                tasks.append(row)
                accepted += 1
    return tasks


def score_response(row, response, *, truncated=False):
    if not isinstance(response, str):
        raise ValueError("response must be text")
    # No eval, arbitrary code, fuzzy substring acceptance or unbounded integer conversion.
    strict = re.fullmatch(r"\s*<answer>(0|[1-9][0-9]{0,8})</answer>\s*", response)
    loose = re.search(r"<answer>(0|[1-9][0-9]{0,8})</answer>", response)
    if loose is None:
        loose = re.fullmatch(r"\s*([0-9]{1,9})\s*", response)
    expected = independent_oracle(row)
    parsed = int(strict.group(1)) if strict else None
    numerical = int(loose.group(1)) if loose else None
    correct = parsed == expected
    return {
        "reward": float(correct and not truncated),
        "strict_correct": bool(correct and not truncated),
        "format_valid": strict is not None,
        "numerical_correct": numerical == expected,
        "parsed": parsed,
        "expected": expected,
        "truncated": bool(truncated),
    }


def training_reward(row, response, *, truncated=False):
    """Training parser/AST label path is separate from evaluation regex/polynomial oracle."""
    text = response.strip()
    if truncated or not text.startswith("<answer>") or not text.endswith("</answer>"):
        return 0.0
    digits = text[8:-9]
    if (
        not 1 <= len(digits) <= 9
        or any(c not in "0123456789" for c in digits)
        or (len(digits) > 1 and digits[0] == "0")
    ):
        return 0.0
    return float(int(digits) == row["answer"])


def wilson_interval(successes, total, z=1.96):
    if (
        type(successes) is not int
        or type(total) is not int
        or not 0 <= successes <= total
        or total < 1
    ):
        raise ValueError("invalid binomial counts")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows):
    if not rows:
        raise ValueError("evaluation requires samples")
    successes = sum(int(r["strict_correct"]) for r in rows)
    return {
        "samples": len(rows),
        "strict_accuracy": successes / len(rows),
        "format_rate": sum(r["format_valid"] for r in rows) / len(rows),
        "numerical_accuracy": sum(r["numerical_correct"] for r in rows) / len(rows),
        "truncation_rate": sum(r["truncated"] for r in rows) / len(rows),
        "wilson_95": wilson_interval(successes, len(rows)),
        "by_family": {
            family: {
                "samples": n,
                "strict_accuracy": sum(r["strict_correct"] for r in rows if r["family"] == family)
                / n,
            }
            for family, n in Counter(r["family"] for r in rows).items()
        },
    }


def dataset_digest(tasks):
    return hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
