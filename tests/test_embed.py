"""Unit tests for chunking + chunk-record building (M5.1).

The embedding pass (`embed_texts`) loads bge-small and is network/model-heavy,
so it is exercised at runtime, not here — these tests cover the pure logic.
"""

import json

from vector import embed


def _paper(pid, title, abstract, has_abstract=True, year=2024, cited=11):
    return {
        "paper_id": pid,
        "title": title,
        "abstract": abstract,
        "has_abstract": has_abstract,
        "text": abstract,
        "year": year,
        "cited_by_count": cited,
        "authors": [],
        "topics": [],
        "references": [],
    }


def test_short_text_is_single_chunk():
    assert embed.chunk_text("just a few words", max_words=256) == ["just a few words"]


def test_empty_text_yields_no_chunks():
    assert embed.chunk_text("   ", max_words=256) == []


def test_long_text_splits_with_overlap():
    words = [f"w{i}" for i in range(100)]
    chunks = embed.chunk_text(" ".join(words), max_words=40, overlap=10)
    # step = 30 -> windows at 0,30,60 (60..100 reaches the end) = 3 chunks
    assert len(chunks) == 3
    first = chunks[0].split()
    second = chunks[1].split()
    assert len(first) == 40
    # last 10 words of chunk 0 == first 10 words of chunk 1 (the overlap)
    assert first[-10:] == second[:10]
    # every original word is covered
    assert set(words) <= set(" ".join(chunks).split())


def test_overlap_must_be_smaller_than_max_words():
    import pytest

    with pytest.raises(ValueError):
        embed.chunk_text("a b c", max_words=10, overlap=10)


def test_build_chunks_joins_title_and_abstract():
    rec = _paper("W1", "My Title", "Some abstract body.")
    chunks = embed.build_chunks([rec])
    assert len(chunks) == 1
    c = chunks[0]
    assert c["chunk_id"] == "W1-0"
    assert c["paper_id"] == "W1"
    assert c["chunk_index"] == 0
    assert c["text"] == "My Title\n\nSome abstract body."
    assert c["year"] == 2024
    assert c["has_abstract"] is True


def test_build_chunks_title_only_when_no_abstract():
    rec = _paper("W2", "Title Only", "", has_abstract=False)
    chunks = embed.build_chunks([rec])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Title Only"


def test_build_chunks_indexes_multi_chunk_paper():
    long_abstract = " ".join(f"x{i}" for i in range(200))
    rec = _paper("W3", "T", long_abstract)
    chunks = embed.build_chunks([rec], max_words=80, overlap=16)
    assert len(chunks) > 1
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert [c["chunk_id"] for c in chunks] == [f"W3-{i}" for i in range(len(chunks))]


def test_write_chunks_roundtrip(tmp_path):
    rec = _paper("W1", "T", "abstract with, a comma")
    chunks = embed.build_chunks([rec])
    out = tmp_path / "chunks.jsonl"
    embed.write_chunks(chunks, str(out))
    back = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert back == chunks
