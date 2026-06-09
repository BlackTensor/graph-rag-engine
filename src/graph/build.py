"""Ingest the cleaned papers into Neo4j as nodes + edges (M4.2).

Reads the nested `data/interim/papers_normalized.jsonl` (M3.2 — the source of
truth that keeps the author->institution nesting the flat CSV collapses) and
MERGEs it into the graph defined in `schema.py`:

  (:Paper)-[:AUTHORED_BY]->(:Author)
  (:Author)-[:WORKS_AT]->(:Institution)
  (:Paper)-[:STUDIES]->(:Topic)
  (:Paper)-[:CITES]->(:Paper)        # in-corpus only (sparse, §10)

Everything is MERGE-based, so re-running is idempotent (no duplicate nodes or
edges). Nodes and the author/topic/institution edges are written in batches;
CITES runs as a final pass once every Paper exists, so a citation to a
later-batch paper still lands. The schema (constraints + indexes) is applied
first so the MERGE keys are index-backed.

    python src/graph/build.py                 # build (idempotent)
    python src/graph/build.py --reset         # DETACH DELETE all nodes, then build
    python src/graph/build.py --batch-size 1000

Reads:  data/interim/papers_normalized.jsonl  (M3.2)
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from graph import schema  # noqa: E402

BATCH_SIZE = 500

# --- Cypher (one statement per concern; each UNWINDs a batch of papers) -------

PAPER_CYPHER = """
UNWIND $papers AS p
MERGE (paper:Paper {paper_id: p.paper_id})
SET paper.title = p.title,
    paper.abstract = p.abstract,
    paper.has_abstract = p.has_abstract,
    paper.year = p.year,
    paper.cited_by_count = p.cited_by_count
"""

AUTHORED_BY_CYPHER = """
UNWIND $papers AS p
UNWIND p.authors AS a
MERGE (author:Author {author_id: a.author_id})
SET author.name = a.name
WITH p, author
MATCH (paper:Paper {paper_id: p.paper_id})
MERGE (paper)-[:AUTHORED_BY]->(author)
"""

WORKS_AT_CYPHER = """
UNWIND $papers AS p
UNWIND p.authors AS a
UNWIND a.institutions AS inst
MERGE (institution:Institution {inst_id: inst.inst_id})
SET institution.name = inst.name,
    institution.ror = inst.ror
WITH a, institution
MATCH (author:Author {author_id: a.author_id})
MERGE (author)-[:WORKS_AT]->(institution)
"""

STUDIES_CYPHER = """
UNWIND $papers AS p
UNWIND p.topics AS t
MERGE (topic:Topic {topic_id: t.topic_id})
SET topic.name = t.name,
    topic.field = t.field
WITH p, topic
MATCH (paper:Paper {paper_id: p.paper_id})
MERGE (paper)-[:STUDIES]->(topic)
"""

# CITES targets are in-corpus, so MATCH (not MERGE) the destination — a stray
# out-of-corpus id simply yields no row instead of creating an empty Paper.
CITES_CYPHER = """
UNWIND $papers AS p
UNWIND p.references AS ref
MATCH (src:Paper {paper_id: p.paper_id})
MATCH (dst:Paper {paper_id: ref})
MERGE (src)-[:CITES]->(dst)
"""

NODE_CYPHER = [PAPER_CYPHER, AUTHORED_BY_CYPHER, WORKS_AT_CYPHER, STUDIES_CYPHER]


def batched(seq: list, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks of seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def reset_graph(session) -> int:
    """DETACH DELETE every node in batches (destructive; gated behind --reset)."""
    total = 0
    while True:
        n = session.run(
            "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS c"
        ).single()["c"]
        total += n
        if n == 0:
            break
    return total


def counts(session) -> dict:
    out = {}
    for label in schema.NODE_KEYS:
        out[label] = session.run(
            f"MATCH (n:{label}) RETURN count(n) AS c"
        ).single()["c"]
    for rtype, _, _ in schema.RELATIONSHIPS:
        out[rtype] = session.run(
            f"MATCH ()-[r:{rtype}]->() RETURN count(r) AS c"
        ).single()["c"]
    return out


def ingest(session, records: Iterable[dict], batch_size: int = BATCH_SIZE) -> dict:
    records = list(records)
    batches = list(batched(records, batch_size))

    # Pass 1: nodes + author/topic/institution edges.
    for batch in batches:
        for cypher in NODE_CYPHER:
            session.run(cypher, papers=batch)

    # Pass 2: CITES, once every Paper node exists.
    for batch in batches:
        session.run(CITES_CYPHER, papers=batch)

    return counts(session)


def main() -> int:
    args = sys.argv[1:]
    do_reset = "--reset" in args
    batch_size = BATCH_SIZE
    if "--batch-size" in args:
        batch_size = int(args[args.index("--batch-size") + 1])

    in_path = os.path.join(config.DATA_INTERIM, "papers_normalized.jsonl")
    if not os.path.exists(in_path):
        print(f"missing {in_path} -- run src/ingest/normalize.py first", file=sys.stderr)
        return 1

    records = load_records(in_path)
    print(f"Loaded {len(records):,} papers from {in_path}")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    try:
        with driver.session() as session:
            schema.apply_schema(session)
            if do_reset:
                deleted = reset_graph(session)
                print(f"--reset: detached/deleted {deleted:,} existing nodes")
            stats = ingest(session, records, batch_size)
    finally:
        driver.close()

    print("Graph built. Counts:")
    for k, v in stats.items():
        print(f"  {k:<14} {v:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
