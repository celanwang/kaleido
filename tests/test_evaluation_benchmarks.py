import json
from pathlib import Path

from kaleido.evaluation.benchmarks import build_suite, gsm_subset
from kaleido.evaluation.schema import freeze_cases

ROOT = Path(__file__).resolve().parents[1]


def test_new_structures_public_subset_are_frozen_before_inference():
    cases, manifest = build_suite(ROOT)
    again, repeated = build_suite(ROOT)
    assert manifest == repeated and freeze_cases(cases) == freeze_cases(again)
    assert manifest["suites"] == {
        "context_qa_v2": 12,
        "gsm8k_fixed_integer_v1": 32,
        "novel_math_v1": 64,
    }
    previous = json.loads((ROOT / "results/p4/reference/tasks.json").read_text())["rows"]
    old = {r["expression"] for r in previous}
    assert all(c.metadata["expression"] not in old for c in cases if c.suite == "novel_math_v1")
    questions, _ = gsm_subset(ROOT / "data/p5")
    raw = [
        json.loads(line)
        for line in (ROOT / "data/p5/raw/gsm8k-test.jsonl").read_text().splitlines()
    ]
    for case in questions:
        assert case.prompt == raw[case.metadata["source_row"]]["question"]
        assert "####" not in case.prompt


def test_color_prompt_and_parser_share_an_explicit_one_word_contract():
    from kaleido.evaluation.scoring import score

    cases, _ = build_suite(ROOT)
    for c in cases:
        if c.suite == "context_qa_v2":
            assert "exactly one color word" in c.system
            assert score(c, c.answer)["correct"]
            assert not score(c, c.answer + " stones")["correct"]
            assert score(c, c.answer + " stones")["content_correct"]
