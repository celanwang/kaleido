import re
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext

from kaleido.posttrain.tasks import wilson_interval

COLORS = ("amber", "azure", "coral", "ivory", "jade", "violet")
NUMBER = r"-?(?:0|[1-9][0-9]{0,31})(?:\.[0-9]{1,16})?"


def canonical_number(text):
    if not isinstance(text, str) or not re.fullmatch(NUMBER, text):
        return None
    try:
        with localcontext() as context:
            context.prec = 64
            value = Decimal(text)
            return "0" if value == 0 else format(value.normalize(), "f")
    except InvalidOperation:
        return None


def score(case, response, *, truncated=False):
    if not isinstance(response, str) or len(response) > 32768:
        raise ValueError("response must be bounded text")
    if case.scorer == "number_xml":
        strict = re.fullmatch(r"\s*<answer>(" + NUMBER + r")</answer>\s*", response)
        matches = re.findall(r"<answer>(" + NUMBER + r")</answer>", response)
        loose = matches[0] if len(matches) == 1 else None
        if not matches:
            bare = re.fullmatch(r"\s*(" + NUMBER + r")\s*", response)
            marker = re.search(r"####\s*(" + NUMBER + r")\s*$", response)
            loose = (bare or marker).group(1) if bare or marker else None
        parsed = canonical_number(strict.group(1)) if strict else None
        numerical = canonical_number(loose) if loose is not None else None
        expected = canonical_number(case.answer)
        if expected is None:
            raise ValueError("numeric case has invalid gold")
        correct = parsed == expected and strict is not None and not truncated
        format_valid = strict is not None
    else:
        if case.answer not in COLORS:
            raise ValueError("context case has invalid gold")
        found = set(re.findall(r"\b(?:" + "|".join(COLORS) + r")\b", response.casefold()))
        color = re.fullmatch(r"\s*(" + "|".join(COLORS) + r")\s*", response.casefold())
        parsed = color.group(1) if color else None
        numerical = next(iter(found)) if len(found) == 1 else None
        expected = case.answer
        correct = parsed == expected and not truncated
        format_valid = parsed is not None
    return {
        "correct": bool(correct),
        "format_valid": bool(format_valid),
        "content_correct": numerical == expected,
        "truncated": bool(truncated),
        "parsed": parsed,
        "expected": expected,
        "failure": "truncated"
        if truncated
        else "unparseable"
        if not format_valid
        else "wrong_answer"
        if not correct
        else None,
    }


def summarize(records):
    if not records:
        raise ValueError("summary requires records")
    successes = sum(int(r["correct"]) for r in records)
    return {
        "samples": len(records),
        "accuracy": successes / len(records),
        "wilson_95": wilson_interval(successes, len(records)),
        "content_accuracy": sum(r["content_correct"] for r in records) / len(records),
        "format_rate": sum(r["format_valid"] for r in records) / len(records),
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
        "failures": dict(Counter(r["failure"] for r in records if r["failure"] is not None)),
    }


def stratified_summary(records):
    return {
        "overall": summarize(records),
        **{
            key: {
                value: summarize([r for r in records if r[key] == value])
                for value in sorted({r[key] for r in records})
            }
            for key in ("suite", "family", "difficulty")
        },
    }
