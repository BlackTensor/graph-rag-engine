"""Unit tests for the Neo4j ingestion (M4.2).

No live Neo4j: these cover the pure helpers (batching, record loading) and lock
the Cypher so every node/edge write stays MERGE-based and idempotent.
"""

import json

from graph import build


def test_batched_chunks_evenly_and_remainder():
    assert list(build.batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(build.batched([], 3)) == []
    assert list(build.batched([1, 2], 5)) == [[1, 2]]


def test_load_records_skips_blank_lines(tmp_path):
    p = tmp_path / "papers.jsonl"
    p.write_text(
        json.dumps({"paper_id": "W1"}) + "\n\n" + json.dumps({"paper_id": "W2"}) + "\n",
        encoding="utf-8",
    )
    recs = build.load_records(str(p))
    assert [r["paper_id"] for r in recs] == ["W1", "W2"]


def test_every_node_and_edge_write_is_merge():
    # MERGE (not CREATE) is what makes re-running the builder idempotent.
    all_cypher = build.NODE_CYPHER + [build.CITES_CYPHER]
    for cypher in all_cypher:
        assert "CREATE (" not in cypher
    assert "MERGE (paper:Paper" in build.PAPER_CYPHER
    assert "MERGE (paper)-[:AUTHORED_BY]->(author)" in build.AUTHORED_BY_CYPHER
    assert "MERGE (author)-[:WORKS_AT]->(institution)" in build.WORKS_AT_CYPHER
    assert "MERGE (paper)-[:STUDIES]->(topic)" in build.STUDIES_CYPHER
    assert "MERGE (src)-[:CITES]->(dst)" in build.CITES_CYPHER


def test_cites_matches_destination_to_avoid_stray_papers():
    # Destination is MATCHed, not MERGEd, so out-of-corpus refs don't create
    # empty Paper nodes.
    assert "MATCH (dst:Paper {paper_id: ref})" in build.CITES_CYPHER
    assert "MERGE (dst:Paper" not in build.CITES_CYPHER
