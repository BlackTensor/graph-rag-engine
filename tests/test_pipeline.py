"""Tests for the end-to-end answer() pipeline (M7.3).

Wiring is verified with the router + LLM monkeypatched (no live services): the
question is routed, the merged context is threaded into the grounded prompt, and
the route/provenance is surfaced alongside the answer. A live end-to-end test
runs the real stack on a simple question and SKIPS when anything is missing
(mirrors tests/test_baseline_e2e.py).
"""

import pytest

import config
from retrieval import pipeline

# --- wiring (no live services) -----------------------------------------------

def test_answer_threads_context_into_prompt_and_returns_provenance(monkeypatch):
    captured = {}

    def fake_route(query):
        return {
            "route": "graph",
            "reason": "relational cue 'collaborat'",
            "executed": ["graph"],
            "fellback": False,
            "merged_context": "Knowledge-graph facts:\nGraph traversal ...",
            "passages": [],
            "graph_result": {"template": "collaborating_institutions", "rows": []},
        }

    def fake_generate(prompt, model=None, host=None, system=pipeline.SYSTEM_PROMPT):
        captured["prompt"] = prompt
        captured["system"] = system
        return "Mostly the Hong Kong universities."

    monkeypatch.setattr(pipeline.router, "route", fake_route)
    monkeypatch.setattr(pipeline.rag, "generate", fake_generate)

    result = pipeline.answer("Which institutions collaborate with Tsinghua University?")

    # The merged context was grounded into the prompt with the unified system prompt.
    assert "Knowledge-graph facts:" in captured["prompt"]
    assert "Question: Which institutions collaborate" in captured["prompt"]
    assert captured["system"] is pipeline.SYSTEM_PROMPT
    # Answer + provenance threaded through.
    assert result["answer"] == "Mostly the Hong Kong universities."
    assert result["route"] == "graph"
    assert result["reason"]
    assert result["executed"] == ["graph"]
    assert "Knowledge-graph facts:" in result["context"]


def test_answer_surfaces_fallback_flag(monkeypatch):
    def fake_route(query):
        return {
            "route": "graph",
            "reason": "relational cue",
            "executed": ["graph(abstained)", "vector(fallback)"],
            "fellback": True,
            "merged_context": "Document passages:\n[1] Some Paper",
            "passages": [{"title": "Some Paper"}],
            "graph_result": {"template": None, "rows": []},
        }

    monkeypatch.setattr(pipeline.router, "route", fake_route)
    monkeypatch.setattr(pipeline.rag, "generate", lambda *a, **k: "answer")

    result = pipeline.answer("Who collaborates with Hogwarts University?")
    assert result["fellback"] is True
    assert result["executed"][-1] == "vector(fallback)"


def test_build_prompt_contains_context_and_question():
    p = pipeline.build_prompt("What is X?", "some context")
    assert "some context" in p and "Question: What is X?" in p


# --- live end-to-end (skips without the stack) -------------------------------

def _require_stack():
    """Skip unless Qdrant (with a built index) and Ollama are both reachable."""
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        coll = config.QDRANT_COLLECTION
        if not client.collection_exists(coll) or client.count(coll, exact=True).count == 0:
            pytest.skip(f"collection '{coll}' missing/empty -- run src/vector/index.py")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 - integration test, skip if down
        pytest.skip(f"Qdrant unreachable ({config.QDRANT_URL}): {exc}")

    import ollama

    try:
        ollama.Client(host=config.OLLAMA_HOST).list()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama unreachable ({config.OLLAMA_HOST}): {exc}")


def test_answer_end_to_end_simple_question():
    _require_stack()
    # A definitional question -> vector route; the pipeline must classify, retrieve,
    # merge, and produce a non-empty grounded answer.
    result = pipeline.answer(
        "Which neural network architecture is based solely on attention mechanisms, "
        "dispensing with recurrence and convolutions?"
    )
    assert result["route"] == "vector"
    assert "Document passages:" in result["context"]
    assert result["answer"].strip(), "empty answer"
