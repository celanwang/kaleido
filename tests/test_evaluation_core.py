import numpy as np
import pytest

from kaleido.evaluation.adapters import ArchiveAdapter
from kaleido.evaluation.runner import evaluate
from kaleido.evaluation.schema import Case, GenerationProtocol, align_predictions, freeze_cases
from kaleido.evaluation.scoring import canonical_number, score
from kaleido.evaluation.statistics import (
    document_nll_bootstrap,
    holm_adjust,
    mcnemar_exact,
    paired_question_bootstrap,
)


def test_frozen_identity_strict_alignment_and_archive_protocol():
    cases = [Case("a", "s", "f", "question", "3")]
    protocol = GenerationProtocol(max_new_tokens=32)
    records = [{"id": "a", "response": "<answer>3</answer>", "truncated": False, "correct": False}]
    adapter = ArchiveAdapter("model", records, source={"revision": "fixed"}, protocol=protocol)
    run = evaluate(adapter, cases, protocol)
    assert run["summary"]["overall"]["accuracy"] == 1
    assert run["identity"]["case_sha256"] == freeze_cases(cases)
    with pytest.raises(ValueError):
        adapter.generate(cases, GenerationProtocol(max_new_tokens=64))
    with pytest.raises(ValueError):
        align_predictions(cases, records + records)
    with pytest.raises(ValueError):
        align_predictions(cases, [])


@pytest.mark.parametrize(
    "response",
    [
        "<answer>3</answer> extra",
        "<answer>03</answer>",
        "<answer>3</answer><answer>3</answer>",
        "<answer>NaN</answer>",
        '__import__("os")',
    ],
)
def test_no_fuzzy_reward_acceptance(response):
    assert not score(Case("a", "s", "f", "q", "3"), response)["correct"]


def test_numeric_equivalence_and_color_oracle_boundaries():
    case = Case("a", "s", "f", "q", "-2.50")
    assert score(case, "<answer>-2.5</answer>")["correct"]
    assert not score(case, "<answer>-2.5</answer>", truncated=True)["correct"]
    assert canonical_number("1e9") is None and canonical_number("-0.0") == "0"
    color = Case("b", "s", "f", "q", "amber", scorer="context_color")
    assert score(color, "The stones are amber.")["correct"]
    assert not score(color, "amber or azure")["correct"]
    assert not score(color, "The stones are not amber.")["correct"]


def test_paired_tests_holm_and_reference_binomial():
    result = mcnemar_exact(np.array([0, 0, 0, 0, 0, 1], bool), np.array([1, 1, 1, 1, 1, 0], bool))
    assert result["wins"] == 5 and result["losses"] == 1
    assert result["exact_two_sided_p"] == 0.21875
    stats = pytest.importorskip("scipy.stats")
    assert result["exact_two_sided_p"] == stats.binomtest(1, 6, p=0.5).pvalue
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2}) == {"a": 0.03, "b": 0.08, "c": 0.2}


def test_clustered_bootstrap_does_not_triple_independent_sample_count():
    before = np.zeros((1, 8))
    after = np.array([[1, 0, 1, 0, 1, 0, 0, 0.0]])
    family = ["a"] * 4 + ["b"] * 4
    one = paired_question_bootstrap(before, after, family, resamples=500, seed=4)
    three = paired_question_bootstrap(
        np.repeat(before, 3, 0), np.repeat(after, 3, 0), family, resamples=500, seed=4
    )
    assert one["percentile_95"] == three["percentile_95"]
    assert three["questions"] == 8 and three["fixed_seeds"] == 3
    assert three == paired_question_bootstrap(
        np.repeat(before, 3, 0), np.repeat(after, 3, 0), family, resamples=500, seed=4
    )


def test_document_nll_is_token_weighted_and_pair_shapes_fail_closed():
    result = document_nll_bootstrap(
        [{"nll_sum": 2.0, "tokens": 2}, {"nll_sum": 27.0, "tokens": 9}], resamples=500
    )
    assert result["nll"] == pytest.approx(29 / 11)
    with pytest.raises(ValueError):
        paired_question_bootstrap(np.zeros((2, 3)), np.ones((2, 2)), ["f"] * 3)


def test_large_decimal_does_not_round_under_default_decimal_context():
    assert canonical_number("12345678901234567890123456789012.1") == (
        "12345678901234567890123456789012.1"
    )
    with pytest.raises(ValueError):
        document_nll_bootstrap([{"tokens": 1, "nll_sum": 1}], resamples=0)
