"""Strict JSON objects for configuration and artifact metadata."""

import json
import math
from pathlib import Path


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("JSON number exceeds finite float range")
    return value


def json_object(raw, *, label="JSON"):
    try:
        value = json.loads(
            raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
        )
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_object(path):
    path = Path(path)
    return json_object(path.read_bytes(), label=str(path))
