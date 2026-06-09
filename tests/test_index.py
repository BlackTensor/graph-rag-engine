"""Tests for the Qdrant index + similarity search (M5.2).

Pure logic (point building, query prefix) runs everywhere; the round-trip
test needs a live Qdrant and is SKIPPED when the service is unreachable, so
`pytest` stays green without Docker (mirrors tests/test_connections.py).
"""

import numpy as np
import pytest

import config
from vector import index, search


def _chunks(n):
    return [
        {
            "chunk_id": f"W{i}-0",
            "paper_id": f"W{i}",
            "chunk_index": 0,
            "title": f"Title {i}",
            "year": 2024,
            "cited_by_count": i,
            "has_abstract": True,
            "text": f"body text {i}",
        }
        for i in range(n)
    ]


def test_make_points_uses_row_index_id_and_keeps_payload():
    chunks = _chunks(3)
    vectors = np.eye(3, dtype="float32")
    points = index.make_points(chunks, vectors)
    assert [p.id for p in points] == [0, 1, 2]
    assert points[0].payload["chunk_id"] == "W0-0"
    assert points[1].vector == [0.0, 1.0, 0.0]


def test_make_points_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        index.make_points(_chunks(3), np.eye(2, dtype="float32"))


def test_query_prefix_is_applied(monkeypatch):
    captured = {}

    class _FakeModel:
        def encode(self, texts, **kwargs):
            captured["text"] = texts[0]
            return np.array([[1.0, 0.0]], dtype="float32")

    monkeypatch.setattr(search, "_get_model", lambda name: _FakeModel())
    vec = search.embed_query("who studies RAG?")
    assert captured["text"].startswith(search.QUERY_PREFIX)
    assert captured["text"].endswith("who studies RAG?")
    assert vec == [1.0, 0.0]


def _qdrant_or_skip():
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        client.get_collections()
        return client
    except Exception as exc:  # noqa: BLE001 - integration test, skip if down
        pytest.skip(f"Qdrant unreachable ({config.QDRANT_URL}): {exc}")


def test_index_and_search_roundtrip():
    client = _qdrant_or_skip()
    collection = "papers_test_m52"
    # 3 orthogonal unit vectors so the nearest neighbour is unambiguous.
    chunks = _chunks(3)
    vectors = np.eye(3, dtype="float32")
    try:
        total = index.index(client, chunks, vectors, collection, batch_size=2, reset=True)
        assert total == 3

        # Re-indexing the same stable ids must not duplicate points (idempotent).
        total_again = index.index(client, chunks, vectors, collection, batch_size=2, reset=False)
        assert total_again == 3

        # Query closest to row 1's vector -> top hit is W1.
        res = client.query_points(
            collection_name=collection, query=[0.0, 1.0, 0.0], limit=1, with_payload=True
        )
        top = res.points[0]
        assert top.payload["paper_id"] == "W1"
        assert top.score == pytest.approx(1.0, abs=1e-5)
    finally:
        client.delete_collection(collection)
