"""Tests for the LangGraph router (M7.1).

The lexical classifier (`classify`) is pure and tested everywhere. The compiled
LangGraph is exercised with the two retrievers monkeypatched, so routing +
the graph->vector hybrid fallback are verified without a live Neo4j/Qdrant.
"""

import pytest

from retrieval import router

# --- classify (pure) ----------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        # relational cues -> graph
        ("Which institutions collaborate most with Tsinghua University?", "graph"),
        ("Who collaborates with Google?", "graph"),
        ("Whose work does Bengio cite?", "graph"),
        ("Which institutions work on the same topics as Ashish Vaswani?", "graph"),
        ("Where does Yoshua Bengio work?", "graph"),
        # entity-enumeration -> graph
        ("Which researchers publish the most here?", "graph"),
        # topic/affiliation intent + named entity -> graph
        ("What does Google research?", "graph"),
        # definitional / semantic -> vector
        ("What is the Transformer architecture?", "vector"),
        ("Explain retrieval-augmented generation.", "vector"),
        ("Summarize the attention mechanism.", "vector"),
        ("What is RAG?", "vector"),
        # topic doc-search without relational structure -> vector
        ("Which papers study reinforcement learning?", "vector"),
    ],
)
def test_classify_route(query, expected):
    assert router.classify(query).route == expected


def test_classify_hybrid_needs_both_signals():
    # relational cue ('cite') + content cue ('explain') -> run both
    d = router.classify("Explain the papers that this work cites.")
    assert d.route == "hybrid"


def test_classify_reason_is_populated():
    assert router.classify("What is attention?").reason
    assert router.classify("Who collaborates with Google?").reason


def test_has_proper_noun():
    assert router._has_proper_noun("What does Google research?")
    assert not router._has_proper_noun("What is attention?")
    # first word is skipped (question openers are capitalized)
    assert not router._has_proper_noun("Which models are best?")


# --- routing through the compiled graph (monkeypatched retrievers) -----------

@pytest.fixture
def fake_retrievers(monkeypatch):
    """Stub graph_retrieve + search so the graph runs with no live services."""
    calls = {"graph": 0, "vector": 0}

    def fake_search(query, k=5, client=None):
        calls["vector"] += 1
        return [{"title": "Some Paper", "text": "...", "score": 0.5}]

    def fake_graph_retrieve(query, **_):
        calls["graph"] += 1
        if "abstain" in query:  # simulate "no entity resolved"
            return {"query": query, "template": None, "seed_type": None,
                    "seed_name": None, "rows": [], "context": "No graph traversal."}
        return {"query": query, "template": "collaborating_institutions",
                "seed_type": "institution", "seed_name": "Tsinghua University",
                "rows": [{"institution": "CUHK", "shared": 13}],
                "context": "Graph traversal ..."}

    monkeypatch.setattr(router.search, "search", fake_search)
    monkeypatch.setattr(router.retrieve, "graph_retrieve", fake_graph_retrieve)
    # rebuild so the cached router (if any) doesn't shadow this run
    monkeypatch.setattr(router, "_router", None)
    return calls


def test_route_vector_question_runs_only_vector(fake_retrievers):
    state = router.route("What is the Transformer architecture?")
    assert state["route"] == "vector"
    assert state["executed"] == ["vector"]
    assert state["vector_contexts"]
    # merge node ran: vector-only context, no graph section
    assert "Document passages:" in state["merged_context"]
    assert "Knowledge-graph facts:" not in state["merged_context"]
    assert fake_retrievers == {"graph": 0, "vector": 1}


def test_route_graph_question_runs_only_graph(fake_retrievers):
    state = router.route("Which institutions collaborate with Tsinghua University?")
    assert state["route"] == "graph"
    assert state["executed"] == ["graph"]
    assert state["graph_result"]["template"] == "collaborating_institutions"
    # merge node ran: graph-only context
    assert "Knowledge-graph facts:" in state["merged_context"]
    assert fake_retrievers == {"graph": 1, "vector": 0}


def test_route_graph_abstain_falls_back_to_vector(fake_retrievers):
    # routed to graph (relational cue) but graph abstains -> vector fallback
    state = router.route("Who collaborates with abstain-org?")
    assert state["route"] == "graph"
    assert state["fellback"] is True
    assert state["executed"] == ["graph(abstained)", "vector(fallback)"]
    assert state["vector_contexts"]
    assert fake_retrievers == {"graph": 1, "vector": 1}


def test_route_hybrid_runs_both(fake_retrievers):
    state = router.route("Explain the papers that Tsinghua University cites.")
    assert state["route"] == "hybrid"
    assert state["executed"] == ["graph", "vector"]
    assert state["graph_result"] and state["vector_contexts"]
    # merge node ran: both sections present in one block
    assert "Knowledge-graph facts:" in state["merged_context"]
    assert "Document passages:" in state["merged_context"]
    assert fake_retrievers == {"graph": 1, "vector": 1}


# --- compiled graph shape -----------------------------------------------------

def test_build_router_compiles():
    compiled = router.build_router()
    assert compiled is not None
    # invoke is the LangGraph runnable entry point
    assert hasattr(compiled, "invoke")
