"""Neo4j node/edge schema + uniqueness constraints and indexes (M4.1).

Defines the property-graph model for the discovery engine and applies the
constraints/indexes that the M4.2 ingestion relies on. Run once against a live
Neo4j (after `docker compose up -d`):

    python src/graph/schema.py            # create constraints + indexes (idempotent)
    python src/graph/schema.py --show     # print existing constraints/indexes
    python src/graph/schema.py --drop      # drop ONLY the schema objects below

Graph model
-----------
Nodes (label : id property : other properties)
  Paper       : paper_id       : title, abstract, has_abstract, year, cited_by_count
  Author      : author_id      : name
  Institution : inst_id        : name, ror
  Topic       : topic_id       : name, field

Relationships (direction matters for traversals)
  (Paper)-[:AUTHORED_BY]->(Author)        a paper was written by an author
  (Author)-[:WORKS_AT]->(Institution)     an author is affiliated with an org
  (Paper)-[:STUDIES]->(Topic)             a paper is about a topic
  (Paper)-[:CITES]->(Paper)               in-corpus citation (sparse, see §10)

The id properties are stable OpenAlex ids (institutions may carry our merged
`ORG:` slug from M3.2); uniqueness constraints make them MERGE keys and create
the backing lookup index for free. Names/year/field get plain indexes for the
filter + lookup patterns the Cypher templates use in M6.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# Node labels and their unique id property (also the MERGE key for ingestion).
NODE_KEYS = {
    "Paper": "paper_id",
    "Author": "author_id",
    "Institution": "inst_id",
    "Topic": "topic_id",
}

# Relationship types: (type, from_label, to_label).
RELATIONSHIPS = [
    ("AUTHORED_BY", "Paper", "Author"),
    ("WORKS_AT", "Author", "Institution"),
    ("STUDIES", "Paper", "Topic"),
    ("CITES", "Paper", "Paper"),
]

# (label, property) pairs to get a plain range index for lookups/filters.
PROPERTY_INDEXES = [
    ("Paper", "year"),
    ("Paper", "cited_by_count"),
    ("Author", "name"),
    ("Institution", "name"),
    ("Topic", "name"),
    ("Topic", "field"),
]

# Full-text index to map natural-language questions -> seed Paper nodes (M6).
FULLTEXT_INDEXES = [
    ("paper_fulltext", "Paper", ["title", "abstract"]),
]


def _constraint_name(label: str) -> str:
    return f"{label.lower()}_id_unique"


def _index_name(label: str, prop: str) -> str:
    return f"{label.lower()}_{prop}_idx"


def constraint_statements() -> list[str]:
    """Idempotent uniqueness constraints for each node's id property."""
    return [
        f"CREATE CONSTRAINT {_constraint_name(label)} IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        for label, key in NODE_KEYS.items()
    ]


def index_statements() -> list[str]:
    """Idempotent range + full-text indexes for lookup/filter patterns."""
    stmts = [
        f"CREATE INDEX {_index_name(label, prop)} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.{prop})"
        for label, prop in PROPERTY_INDEXES
    ]
    for name, label, props in FULLTEXT_INDEXES:
        each = ", ".join(f"n.{p}" for p in props)
        stmts.append(
            f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON EACH [{each}]"
        )
    return stmts


def drop_statements() -> list[str]:
    """Drop only the schema objects this module manages (data is untouched)."""
    drops = [
        f"DROP CONSTRAINT {_constraint_name(label)} IF EXISTS" for label in NODE_KEYS
    ]
    drops += [
        f"DROP INDEX {_index_name(label, prop)} IF EXISTS"
        for label, prop in PROPERTY_INDEXES
    ]
    drops += [f"DROP INDEX {name} IF EXISTS" for name, _, _ in FULLTEXT_INDEXES]
    return drops


def apply_schema(session) -> dict:
    """Run every constraint + index statement against an open Neo4j session."""
    constraints = constraint_statements()
    indexes = index_statements()
    for stmt in constraints + indexes:
        session.run(stmt)
    return {"constraints": len(constraints), "indexes": len(indexes)}


def drop_schema(session) -> int:
    stmts = drop_statements()
    for stmt in stmts:
        session.run(stmt)
    return len(stmts)


def show_schema(session) -> list[str]:
    lines = []
    for row in session.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties"):
        lines.append(
            f"CONSTRAINT {row['name']} {row['type']} "
            f"{row['labelsOrTypes']} {row['properties']}"
        )
    for row in session.run("SHOW INDEXES YIELD name, type, labelsOrTypes, properties"):
        lines.append(
            f"INDEX      {row['name']} {row['type']} "
            f"{row['labelsOrTypes']} {row['properties']}"
        )
    return lines


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    driver = _driver()
    try:
        with driver.session() as session:
            if mode == "--show":
                for line in show_schema(session):
                    print(line)
            elif mode == "--drop":
                n = drop_schema(session)
                print(f"Dropped {n} schema objects (data untouched).")
            elif mode == "--apply":
                stats = apply_schema(session)
                print(
                    f"Applied {stats['constraints']} constraints + "
                    f"{stats['indexes']} indexes."
                )
                for line in show_schema(session):
                    print("  " + line)
            else:
                print(f"unknown mode {mode!r} (use --apply/--show/--drop)", file=sys.stderr)
                return 2
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
