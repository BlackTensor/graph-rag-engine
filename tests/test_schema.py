"""Unit tests for the Neo4j schema definition (M4.1).

These are pure-statement tests -- no live Neo4j required. They lock the graph
model (labels, id keys, relationships) and assert every DDL statement is
idempotent so re-running the builder is always safe.
"""

from graph import schema


def test_node_keys_match_entity_model():
    assert schema.NODE_KEYS == {
        "Paper": "paper_id",
        "Author": "author_id",
        "Institution": "inst_id",
        "Topic": "topic_id",
    }


def test_relationships_cover_the_four_edge_types():
    types = {r[0] for r in schema.RELATIONSHIPS}
    assert types == {"AUTHORED_BY", "WORKS_AT", "STUDIES", "CITES"}
    # CITES is the only self-referential (Paper -> Paper) edge.
    assert ("CITES", "Paper", "Paper") in schema.RELATIONSHIPS


def test_one_unique_constraint_per_node_label():
    stmts = schema.constraint_statements()
    assert len(stmts) == len(schema.NODE_KEYS)
    for (label, key), stmt in zip(schema.NODE_KEYS.items(), stmts):
        assert f"(n:{label})" in stmt
        assert f"n.{key} IS UNIQUE" in stmt


def test_all_ddl_is_idempotent():
    for stmt in schema.constraint_statements() + schema.index_statements():
        assert "IF NOT EXISTS" in stmt
    for stmt in schema.drop_statements():
        assert "IF EXISTS" in stmt


def test_indexes_cover_lookup_properties_and_fulltext():
    stmts = schema.index_statements()
    joined = "\n".join(stmts)
    assert "FOR (n:Paper) ON (n.year)" in joined
    assert "FOR (n:Topic) ON (n.field)" in joined
    # Full-text over title + abstract for NL -> seed-node mapping (M6).
    assert any("FULLTEXT INDEX paper_fulltext" in s and "n.title" in s for s in stmts)
