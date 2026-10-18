"""Predeclared held-out structures, pinned public subset and reference-bounded QA."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from kaleido.evaluation.schema import Case, digest, freeze_cases
from kaleido.evaluation.scoring import COLORS, canonical_number
from kaleido.posttrain.tasks import SYSTEM
from kaleido.posttrain.tasks import expression as old_expression


def signature(node):
    return "n" if type(node) is int else f"{node[0]}({signature(node[1])},{signature(node[2])})"


def _value(node):
    if type(node) is int:
        return node
    op, x, y = node
    a, b = _value(x), _value(y)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "mod":
        return a % b
    raise ValueError("unknown benchmark operator")


def _render(node):
    if type(node) is int:
        return str(node)
    symbols = {"add": "+", "sub": "-", "mul": "*", "mod": "%"}
    return f"({_render(node[1])} {symbols[node[0]]} {_render(node[2])})"


def novel_math(*, seed=501, per_family=16, previous_rows=()):
    if type(per_family) is not int or per_family < 2 or per_family % 2:
        raise ValueError("novel-math count per family must be positive and even")
    rng = np.random.default_rng(seed)
    seen = set()
    cases = []
    previous_shapes = {
        signature(old_expression(r["family"], **r["operands"])) for r in previous_rows
    }
    previous_formulas = {r["expression"] for r in previous_rows}
    families = ("difference_product", "shifted_product", "product_of_differences", "nested_offset")
    for family in families:
        accepted = 0
        for _attempt in range(per_family * 10000):
            if accepted == per_family:
                break
            low, high = (1, 31) if accepted < per_family // 2 else (31, 61)
            a, b, c, d = map(int, rng.integers(low, high, 4))
            m = int(rng.integers(8, 18))
            trees = {
                "difference_product": ("mul", ("sub", a, b), ("add", c, d)),
                "shifted_product": ("sub", ("mul", ("add", a, b), c), d),
                "product_of_differences": ("mul", ("sub", a, b), ("sub", c, d)),
                "nested_offset": ("add", a, ("mul", b, ("sub", c, d))),
            }
            expanded = {
                "difference_product": a * c + a * d - b * c - b * d,
                "shifted_product": a * c + b * c - d,
                "product_of_differences": a * c - a * d - b * c + b * d,
                "nested_offset": a + b * c - b * d,
            }[family]
            node = ("mod", trees[family], m)
            answer = _value(node)
            if answer != expanded - m * (expanded // m):
                raise AssertionError("new independent oracles disagree")
            if answer != accepted % 8:
                continue
            formula = _render(node)
            if formula in previous_formulas or formula in seen:
                continue
            shape = signature(node)
            if shape in previous_shapes:
                raise AssertionError("new AST family leaked into P4")
            seen.add(formula)
            cases.append(
                Case(
                    id="novel-" + hashlib.sha256(formula.encode()).hexdigest()[:20],
                    suite="novel_math_v1",
                    family=family,
                    prompt=(
                        f"Compute {formula}. The symbol % means the least non-negative remainder."
                    ),
                    answer=str(answer),
                    system=SYSTEM,
                    difficulty="small_operands" if low == 1 else "large_operands",
                    metadata={
                        "expression": formula,
                        "signature": shape,
                        "operands": [a, b, c, d, m],
                    },
                )
            )
            accepted += 1
        if accepted != per_family:
            raise ValueError("not enough novel balanced expressions")
    return cases


def gsm_subset(directory, *, seed=1701, count=32):
    directory = Path(directory)
    source = json.loads((directory / "sources.json").read_text())
    for name, info in source["files"].items():
        if hashlib.sha256((directory / "raw" / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("public dataset source checksum changed")
    rows = [
        json.loads(line) for line in (directory / "raw/gsm8k-test.jsonl").read_text().splitlines()
    ]
    eligible = []
    for i, row in enumerate(rows):
        final = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        number = canonical_number(final)
        if number is not None and re.fullmatch(r"-?[0-9]+", number):
            eligible.append((i, row, number))
    if type(count) is not int or not 1 <= count <= len(eligible):
        raise ValueError("invalid public subset size")
    chosen = np.random.default_rng(seed).choice(len(eligible), size=count, replace=False)
    cases = []
    for index in sorted(map(int, chosen)):
        row_index, row, answer = eligible[index]
        cases.append(
            Case(
                id=f"gsm8k-test-{row_index:04d}",
                suite="gsm8k_fixed_integer_v1",
                family="word_problem",
                prompt=row["question"],
                answer=answer,
                difficulty="short_prompt" if len(row["question"]) < 250 else "long_prompt",
                metadata={
                    "source_row": row_index,
                    "source_revision": source["revision"],
                    "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                },
            )
        )
    return cases, {
        "source_rows": len(rows),
        "eligible_integer_rows": len(eligible),
        "selection_seed": seed,
        "selected_source_rows": [c.metadata["source_row"] for c in cases],
        "source": source,
        "note": (
            "custom integer/direct-answer subset, not full-GSM8K CoT; pretraining overlap unknown"
        ),
    }


def context_cases(*, count=12, prefix="context", seed=590):
    if type(count) is not int or count < 1:
        raise ValueError("context suite count must be positive")
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(count):
        first, second = rng.choice(COLORS, 2, replace=False).tolist()
        a, b = f"{prefix}-Nera-{i}", f"{prefix}-Tova-{i}"
        text = f"Record: room {a} contains {first} stones; room {b} contains {second} stones."
        cases.append(
            Case(
                id=f"{prefix}-{i:03d}",
                suite="context_qa_v1",
                family="color_retrieval",
                prompt=f"{text}\nQuestion: What color are the stones in room {a}?",
                answer=first,
                scorer="context_color",
                system="Answer concisely using only the supplied record.",
                difficulty="bounded_reference",
                metadata={"context": text, "distractor": second},
            )
        )
    return cases


def build_suite(root, *, math_per_family=16, gsm_count=32, qa_count=12, seed=1701):
    root = Path(root)
    previous = json.loads((root / "results/p4/reference/tasks.json").read_text())["rows"]
    math_cases = novel_math(seed=seed + 1, per_family=math_per_family, previous_rows=previous)
    public, source = gsm_subset(root / "data/p5", seed=seed, count=gsm_count)
    qa = context_cases(count=qa_count, seed=seed + 2)
    cases = math_cases + public + qa
    return cases, {
        "schema_version": 1,
        "case_sha256": freeze_cases(cases),
        "selection_seed": seed,
        "prior_p4_dataset_sha256": digest(previous),
        "public_subset": source,
        "suites": {
            name: len([c for c in cases if c.suite == name])
            for name in sorted({c.suite for c in cases})
        },
        "limitations": [
            "controlled label-balanced novel expressions",
            "same primitive arithmetic still overlaps",
            "GSM8K pretrained overlap not measurable here",
            "reference-color matching is a narrow oracle, not full factuality",
        ],
    }
