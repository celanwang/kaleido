import pytest

from kaleido.evaluation.judge import calibration_pairs, judge_cases, summarize_judgments


def predictions(pairs):
    rows = []
    for p in pairs:
        for swapped in (False, True):
            value = (
                "TIE" if p["gold"] == "tie" else ("A" if (p["gold"] == "left") != swapped else "B")
            )
            rows.append(
                {
                    "id": p["id"] + ("-swap" if swapped else "-normal"),
                    "response": value,
                    "truncated": False,
                }
            )
    return rows


def test_judge_order_mapping_ties_and_invalid_denominators():
    pairs = calibration_pairs()
    rows = predictions(pairs)
    result = summarize_judgments(pairs, rows)
    assert result["summary"]["agreement"] == result["summary"]["swap_consistency"] == 1
    assert result["summary"]["passes"]
    rows[0]["response"] = "A because it is correct"
    rows[1]["truncated"] = True
    result = summarize_judgments(pairs, rows)
    assert result["summary"]["agreement"] == 30 / 32
    assert result["summary"]["parse_rate"] == 30 / 32
    assert not result["summary"]["passes"]
    with pytest.raises(ValueError):
        summarize_judgments(pairs, rows[:-1])


def test_always_A_judge_fails_and_audit_is_disjoint():
    pairs = calibration_pairs()
    rows = [{"id": c.id, "response": "A", "truncated": False} for c in judge_cases(pairs)]
    summary = summarize_judgments(pairs, rows)["summary"]
    assert summary["swap_consistency"] == 0 and not summary["passes"]
    audit = calibration_pairs(split="audit")
    assert not {p["question"] for p in pairs} & {p["question"] for p in audit}
    assert {p["category"] for p in pairs} == {"direct", "verbosity", "injection", "tie"}
