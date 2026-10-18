"""Deterministic byte BPE with a fixed regex partition and explicit EOS."""

import json
import re
from collections import Counter
from pathlib import Path

from kaleido.experiments import write_json

PATTERN = r"\s+|\w+|[^\w\s]+"
EOS = 256


def _merge(ids, pair, new_id):
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return tuple(result)


class ByteBPE:
    def __init__(self, merges=()):
        self.merges = []
        self.pieces = [bytes([i]) for i in range(256)] + [b""]
        self.ranks = {}
        for pair in merges:
            pair = tuple(pair)
            new_id = len(self.pieces)
            if (
                len(pair) != 2
                or any(type(i) is not int or i < 0 or i >= new_id or i == EOS for i in pair)
                or pair in self.ranks
            ):
                raise ValueError("invalid or duplicate BPE merge")
            self.ranks[pair] = new_id
            self.merges.append(pair)
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])

    def __len__(self):
        return len(self.pieces)

    @classmethod
    def fit(cls, documents, *, vocab_size=512):
        if type(vocab_size) is not int or vocab_size < EOS + 1:
            raise ValueError("vocab_size must be an integer >=257")
        words = Counter()
        for text in documents:
            if not isinstance(text, str):
                raise ValueError("BPE training documents must be strings")
            words.update(tuple(chunk.encode("utf-8")) for chunk in re.findall(PATTERN, text))
        if not words:
            raise ValueError("BPE requires non-empty training text")
        merges = []
        for new_id in range(EOS + 1, vocab_size):
            counts = Counter()
            for word, frequency in words.items():
                for pair in zip(word[:-1], word[1:], strict=True):
                    counts[pair] += frequency
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            merges.append(pair)
            updated = Counter()
            for word, frequency in words.items():
                updated[_merge(word, pair, new_id)] += frequency
            words = updated
        return cls(merges)

    def encode(self, text, *, add_eos=False):
        if not isinstance(text, str):
            raise ValueError("encode requires a Unicode string")
        result = []
        for chunk in re.findall(PATTERN, text):
            ids = tuple(chunk.encode("utf-8"))
            while len(ids) > 1:
                pairs = list(zip(ids[:-1], ids[1:], strict=True))
                pair = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
                if pair not in self.ranks:
                    break
                ids = _merge(ids, pair, self.ranks[pair])
            result.extend(ids)
        return result + [EOS] if add_eos else result

    def decode(self, ids, *, errors="strict"):
        values = []
        for token in ids:
            if type(token) is not int or not 0 <= token < len(self):
                raise ValueError("token ID out of range")
            values.append(self.pieces[token])
        return b"".join(values).decode("utf-8", errors=errors)

    def to_dict(self):
        return {
            "version": 1,
            "pattern": PATTERN,
            "eos_id": EOS,
            "vocab_size": len(self),
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, path):
        write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            set(value) != {"version", "pattern", "eos_id", "vocab_size", "merges"}
            or value["version"] != 1
            or value["pattern"] != PATTERN
            or value["eos_id"] != EOS
        ):
            raise ValueError("unsupported tokenizer format")
        tokenizer = cls(value["merges"])
        if value["vocab_size"] != len(tokenizer):
            raise ValueError("vocabulary length mismatch")
        return tokenizer
