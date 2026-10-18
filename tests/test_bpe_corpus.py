import json
from pathlib import Path

import pytest

from kaleido.lm.corpus import chapter_documents, prepare_corpus
from kaleido.lm.tokenizer import EOS, ByteBPE

ROOT = Path(__file__).resolve().parents[1]


def test_byte_bpe_roundtrip_determinism_and_no_special_injection(tmp_path):
    texts = ["banana banana banana\n", "你好 café 👋\t<|eos|> 123!"]
    a, b = ByteBPE.fit(texts, vocab_size=290), ByteBPE.fit(reversed(texts), vocab_size=290)
    assert a.to_dict() == b.to_dict()
    for text in [*texts, "未见过的🙂\x00\r\n", "", "a" * 100]:
        assert a.decode(a.encode(text)) == text
        assert EOS not in a.encode(text)
        assert a.decode(a.encode(text, add_eos=True)) == text
    path = tmp_path / "bpe.json"
    a.save(path)
    assert ByteBPE.load(path).to_dict() == a.to_dict()
    assert len(a.encode("banana banana")) < len(b"banana banana")


def test_known_nonoverlap_merges_and_strict_validation(tmp_path):
    tokenizer = ByteBPE([(97, 97), (257, 257)])
    assert tokenizer.encode("aaaaa") == [258, 97]
    for merges in ([(256, 0)], [(300, 0)], [(97, 97), (97, 97)]):
        with pytest.raises(ValueError):
            ByteBPE(merges)
    with pytest.raises(ValueError):
        tokenizer.decode([999])
    with pytest.raises(ValueError):
        ByteBPE.fit([])
    path = tmp_path / "broken.json"
    tokenizer.save(path)
    value = json.loads(path.read_text())
    value["vocab_size"] = 999
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        ByteBPE.load(path)


def test_pinned_corpus_chapter_splits_and_overlap_audit():
    documents, audit = prepare_corpus(ROOT / "data/p3")
    again, repeated = prepare_corpus(ROOT / "data/p3")
    assert documents == again and audit == repeated
    assert len(documents) == 48
    assert [len(audit["splits"][s]["documents"]) for s in ("train", "valid", "test")] == [40, 4, 4]
    assert len({d["sha256"] for d in documents}) == 48
    seen = {}
    for doc in documents:
        assert "PROJECT GUTENBERG LICENSE" not in doc["text"]
        for paragraph in doc["paragraphs"]:
            if len(paragraph) >= 80:
                assert paragraph.casefold() not in seen
                seen[paragraph.casefold()] = doc["id"]
    assert audit["near_overlap"]["jaccard"] < 0.8


def test_source_and_document_boundaries_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        chapter_documents(b"Chapter I\ntext", "x", 1)
    (tmp_path / "raw").mkdir()
    (tmp_path / "sources.json").write_text(json.dumps({"books": [{"id": "x", "sha256": "wrong"}]}))
    (tmp_path / "raw/x.txt").write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        prepare_corpus(tmp_path)
