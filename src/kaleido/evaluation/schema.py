import hashlib
import json
from dataclasses import asdict, dataclass, field


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Case:
    id: str
    suite: str
    family: str
    prompt: str
    answer: str
    scorer: str = "number_xml"
    system: str = (
        "Return exactly one numeric answer in <answer>NUMBER</answer>. Do not include explanation."
    )
    split: str = "test"
    difficulty: str = "unspecified"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if any(
            not isinstance(v, str) or not v
            for v in (self.id, self.suite, self.family, self.prompt, self.answer, self.system)
        ):
            raise ValueError("case identifiers, prompt, answer and system must be non-empty text")
        if self.scorer not in ("number_xml", "context_color") or self.split not in (
            "train",
            "valid",
            "test",
            "calibration",
            "audit",
        ):
            raise ValueError("unsupported scorer/split")
        digest(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GenerationProtocol:
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int = 1701
    batch_size: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_new_tokens, self.batch_size)):
            raise ValueError("generation token/batch limits must be positive integers")
        if type(self.seed) is not int or self.seed < 0 or self.temperature != 0:
            raise ValueError("P5 freezes greedy decoding and a non-negative seed")

    def to_dict(self):
        return asdict(self)


def freeze_cases(cases):
    cases = list(cases)
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("cases must be non-empty with unique IDs")
    return digest([c.to_dict() for c in cases])


def align_predictions(cases, records):
    cases = list(cases)
    records = list(records)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate prediction IDs")
    mapping = {r["id"]: r for r in records}
    if set(mapping) != {c.id for c in cases}:
        raise ValueError("predictions must cover exactly the frozen case IDs")
    return [mapping[c.id] for c in cases]
