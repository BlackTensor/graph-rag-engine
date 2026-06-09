"""End-to-end confirmation of the vector-RAG baseline (M5.4).

Runs the full pipeline — Qdrant retrieval + Ollama generation — on one simple,
well-covered question and confirms it answers correctly. Requires the live
stack AND a built index (M5.2) AND a pulled model (M5.3); if any piece is
missing the test SKIPS rather than fails, so `pytest` stays green without the
full local setup (mirrors tests/test_connections.py).
"""

import pytest

import config
from vector import rag

# A content-descriptive question that the corpus answers unambiguously: the
# abstract of "Attention Is All You Need" is its rank-1 retrieval.
QUESTION = (
    "Which neural network architecture is based solely on attention mechanisms, "
    "dispensing with recurrence and convolutions?"
)
EXPECTED_PAPER = "Attention Is All You Need"


def _client_or_skip():
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        coll = config.QDRANT_COLLECTION
        if not client.collection_exists(coll):
            pytest.skip(f"collection '{coll}' missing -- run src/vector/index.py")
        if client.count(coll, exact=True).count == 0:
            pytest.skip(f"collection '{coll}' empty -- run src/vector/index.py")
        return client
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 - integration test, skip if down
        pytest.skip(f"Qdrant unreachable ({config.QDRANT_URL}): {exc}")


def _require_ollama():
    import ollama

    try:
        ollama.Client(host=config.OLLAMA_HOST).list()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama unreachable ({config.OLLAMA_HOST}): {exc}")


def test_baseline_answers_simple_question_end_to_end():
    client = _client_or_skip()
    _require_ollama()

    result = rag.vector_rag(QUESTION, k=4, client=client)

    # Retrieval (deterministic) must surface the right paper among the contexts.
    titles = [c.get("title", "") for c in result["contexts"]]
    assert EXPECTED_PAPER in titles, titles

    # Generation must produce a grounded, on-topic answer (not the "don't know").
    answer = result["answer"].lower()
    assert answer.strip(), "empty answer"
    assert "transformer" in answer or "attention" in answer, result["answer"]
