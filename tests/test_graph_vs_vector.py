"""Verify Graph-RAG answers a multi-hop question vector RAG can't (M6.3).

Pure wiring (prompt assembly, graph_retrieve->generate) runs everywhere via
monkeypatch. The headline contrast is an integration test that needs both
Neo4j AND Qdrant and is SKIPPED when either is down; it asserts the contrast at
the *retrieval* level (deterministic), not on LLM wording:

  * Graph-RAG resolves the seed, routes to the collaboration traversal, and
    returns a ranked list of institutions (the answer).
  * The vector baseline's retrieved chunks (title+abstract only) do NOT contain
    that answer — affiliation/collaboration data was never embedded — so the
    baseline has nothing to ground a correct answer on.
"""

import pytest

import config
from graph import compare, retrieve
from vector import search

# --- pure wiring --------------------------------------------------------------

def test_build_graph_prompt_includes_context_and_question():
    prompt = compare.build_graph_prompt("Who collaborates with X?", "  1. institution=Y")
    assert "Graph results:" in prompt
    assert "institution=Y" in prompt
    assert "Question: Who collaborates with X?" in prompt


def test_graph_system_prompt_constrains_to_graph_results():
    assert "ONLY" in compare.GRAPH_SYSTEM_PROMPT
    assert "don't know" in compare.GRAPH_SYSTEM_PROMPT


def test_graph_rag_wires_retrieve_to_generate(monkeypatch):
    retrieved = {
        "query": "Q", "seed_type": "institution", "seed_name": "Tsinghua University",
        "template": "collaborating_institutions",
        "params": {"inst_name": "Tsinghua University"},
        "rows": [{"institution": "University of Hong Kong", "shared_papers": 14}],
        "context": "Graph traversal `collaborating_institutions`\n  1. institution=...",
    }
    captured = {}

    monkeypatch.setattr(
        compare.retrieve, "graph_retrieve",
        lambda query, session=None, limit=15: dict(retrieved, query=query),
    )

    def fake_generate(prompt, model=None, host=None, system=None):
        captured["prompt"] = prompt
        captured["system"] = system
        return "Top collaborator: University of Hong Kong (14 papers)."

    monkeypatch.setattr(compare.rag, "generate", fake_generate)

    result = compare.graph_rag("Who collaborates with Tsinghua University?")
    assert result["template"] == "collaborating_institutions"
    assert result["answer"].startswith("Top collaborator")
    # the traversal context must have reached the prompt + the graph system prompt used
    assert "collaborating_institutions" in captured["prompt"]
    assert captured["system"] is compare.GRAPH_SYSTEM_PROMPT


# --- live contrast (skips without Neo4j + Qdrant) -----------------------------

def _neo4j_or_skip():
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        return driver
    except Exception as exc:  # noqa: BLE001 - integration test, skip if down
        pytest.skip(f"Neo4j unreachable ({config.NEO4J_URI}): {exc}")


def _qdrant_or_skip():
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        client.get_collections()
        return client
    except Exception as exc:  # noqa: BLE001 - integration test, skip if down
        pytest.skip(f"Qdrant unreachable ({config.QDRANT_URL}): {exc}")


def test_multihop_graph_answers_what_vector_cannot():
    driver = _neo4j_or_skip()
    qdrant = _qdrant_or_skip()
    query = compare.DEMO_QUESTION  # collaboration question

    try:
        with driver.session() as session:
            g = retrieve.graph_retrieve(query, session=session, limit=10)
    finally:
        driver.close()

    # Graph-RAG produces the answer: a ranked list of collaborating institutions.
    assert g["seed_type"] == "institution"
    assert g["template"] == "collaborating_institutions"
    assert g["rows"], "graph should return collaborating institutions"
    answer_insts = {r["institution"] for r in g["rows"]}

    # Vector baseline: the retrieved title+abstract chunks don't contain the
    # affiliation/collaboration answer, so it can't ground a correct response.
    hits = search.search(query, k=5, client=qdrant)
    chunk_blob = " ".join(h.get("text", "") for h in hits).lower()
    present = [name for name in answer_insts if name.lower() in chunk_blob]
    assert len(present) <= 1, (
        f"vector chunks unexpectedly contained the answer institutions: {present}"
    )
