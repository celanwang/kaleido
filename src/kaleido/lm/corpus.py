"""Pinned sources -> chapter documents -> deterministic split and overlap audit."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def chapter_documents(raw, book_id, expected_chapters):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    starts = list(re.finditer(r"^\*\*\* START OF .+?\*\*\*$", text, re.M))
    ends = list(re.finditer(r"^\*\*\* END OF .+?\*\*\*$", text, re.M))
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        raise ValueError("source must have one complete Gutenberg body boundary")
    body = text[starts[0].end() : ends[0].start()]
    headings = list(re.finditer(r"^CHAPTER [IVXLCDM]+\.?$", body, re.M | re.I))
    if len(headings) != expected_chapters:
        raise ValueError(f"{book_id}: chapter count changed; inspect the pinned source")
    documents = []
    for i, heading in enumerate(headings):
        segment = body[
            heading.end() : headings[i + 1].start() if i + 1 < len(headings) else len(body)
        ].strip()
        title, content = segment.split("\n\n", 1)
        paragraphs = [
            unicodedata.normalize("NFC", re.sub(r"\s+", " ", p).strip())
            for p in re.split(r"\n\s*\n", content)
        ]
        paragraphs = [p for p in paragraphs if p]
        if not paragraphs:
            raise ValueError("empty chapter")
        documents.append(
            {
                "id": f"{book_id}-chapter-{i + 1:02d}",
                "book": book_id,
                "title": title.strip(),
                "paragraphs": paragraphs,
            }
        )
    return documents


def prepare_corpus(directory, *, split_seed=17):
    directory = Path(directory)
    sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    documents = []
    for book in sources["books"]:
        raw = (directory / "raw" / (book["id"] + ".txt")).read_bytes()
        if sha256(raw) != book["sha256"]:
            raise ValueError(f"{book['id']}: raw source digest mismatch")
        documents.extend(chapter_documents(raw, book["id"], book["chapters"]))
    seen_documents, seen_paragraphs = set(), set()
    cleaned = []
    duplicate_paragraphs = 0
    duplicate_documents = 0
    for document in documents:
        original = "\n\n".join(document["paragraphs"])
        fingerprint = sha256(original.encode())
        if fingerprint in seen_documents:
            duplicate_documents += 1
            continue
        seen_documents.add(fingerprint)
        paragraphs = []
        for paragraph in document["paragraphs"]:
            canonical = re.sub(r"\s+", " ", paragraph.casefold()).strip()
            if len(canonical) >= 80:
                if canonical in seen_paragraphs:
                    duplicate_paragraphs += 1
                    continue
                seen_paragraphs.add(canonical)
            paragraphs.append(paragraph)
        if not paragraphs:
            raise ValueError("deduplication removed an entire unique chapter")
        text = "\n\n".join(paragraphs)
        cleaned.append(
            {**document, "paragraphs": paragraphs, "text": text, "sha256": sha256(text.encode())}
        )
    for book in sources["books"]:
        group = [d for d in cleaned if d["book"] == book["id"]]
        group.sort(key=lambda d: sha256(f"{split_seed}:{d['id']}".encode()))
        held = max(1, len(group) // 10)
        if len(group) <= 2 * held:
            raise ValueError("not enough independent chapters for three splits")
        for i, doc in enumerate(group):
            doc["split"] = "valid" if i < held else "test" if i < 2 * held else "train"
    # Near-overlap audit is measured, not asserted absent based only on exact hashes.
    shingles = {
        d["id"]: set(
            zip(*(re.findall(r"\w+", d["text"].casefold())[i:] for i in range(5)), strict=False)
        )
        for d in cleaned
    }
    largest = {"jaccard": 0.0, "document_ids": []}
    for i, a in enumerate(cleaned):
        for b in cleaned[i + 1 :]:
            if a["split"] == b["split"]:
                continue
            sa, sb = shingles[a["id"]], shingles[b["id"]]
            score = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
            if score > largest["jaccard"]:
                largest = {"jaccard": score, "document_ids": [a["id"], b["id"]]}
    if largest["jaccard"] >= 0.8:
        raise ValueError("cross-split near-duplicate chapters require grouping before training")
    audit = {
        "source_sha256": sha256((directory / "sources.json").read_bytes()),
        "split_seed": split_seed,
        "duplicate_documents_removed": duplicate_documents,
        "duplicate_long_paragraphs_removed": duplicate_paragraphs,
        "long_paragraph_min_chars": 80,
        "near_overlap": largest,
        "near_duplicate_threshold": 0.8,
        "splits": {
            name: {
                "documents": [d["id"] for d in cleaned if d["split"] == name],
                "utf8_bytes": sum(len(d["text"].encode()) for d in cleaned if d["split"] == name),
            }
            for name in ("train", "valid", "test")
        },
    }
    return cleaned, audit
