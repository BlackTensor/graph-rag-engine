"""Tests for the baseline vector-RAG pipeline (M5.3, M8.2).

Passage rendering and the retrieve->ground->answer wiring are covered with pure
logic / monkeypatching (no Qdrant or Ollama). As of M8.2 the LLM call is the
shared writer in `llm/client.py`; here we assert vector_rag threads the
retrieved passages into that writer. The live end-to-end run is M5.4.
"""

from vector import rag


def test_format_context_numbers_passages():
    ctx = rag.format_context(
        [
            {"title": "Paper A", "text": "alpha"},
            {"title": "Paper B", "text": "beta"},
        ]
    )
    assert "[1] Paper A\nalpha" in ctx
    assert "[2] Paper B\nbeta" in ctx


def test_vector_rag_wires_retrieval_into_shared_writer(monkeypatch):
    hits = [
        {"title": "Attention Is All You Need", "text": "transformer", "score": 0.9,
         "paper_id": "W1", "chunk_index": 0},
    ]
    captured = {}

    monkeypatch.setattr(rag.search, "search", lambda query, k, client=None: hits)

    def fake_answer(query, context, model=None, host=None):
        captured["query"] = query
        captured["context"] = context
        return "Grounded answer."

    # M8.2: vector_rag writes through llm/client.py, not a local generate().
    monkeypatch.setattr(rag.llm_client, "answer_from_context", fake_answer)

    result = rag.vector_rag("What is the Transformer?", k=3)
    assert result["query"] == "What is the Transformer?"
    assert result["answer"] == "Grounded answer."
    assert result["contexts"] is hits
    # the retrieved chunk text must have reached the context the writer saw
    assert "transformer" in captured["context"]
    assert "Attention Is All You Need" in captured["context"]
    assert captured["query"] == "What is the Transformer?"
