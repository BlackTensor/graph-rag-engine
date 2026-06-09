"""Verify the built graph: counts, integrity, spot-check traversals (M4.3).

Independently recomputes the expected node/edge cardinalities straight from
`papers_normalized.jsonl` (distinct sets, same MERGE semantics as the builder)
and compares them against the live Neo4j. Then runs a few integrity checks and
the kind of multi-hop traversals you'd eyeball in Neo4j Browser, so the build is
verified end-to-end without a human clicking around.

    python src/graph/verify.py        # exits 0 if every count matches, 1 otherwise

The SAMPLE_QUERIES at the bottom are printed for copy-paste into Neo4j Browser
(http://localhost:7474) to capture the M4.4 screenshots.

Reads:  data/interim/papers_normalized.jsonl  (M3.2)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# Cypher you can paste into Neo4j Browser for the M4.4 screenshots.
SAMPLE_QUERIES = {
    "schema": "CALL db.schema.visualization()",
    "transformer_neighbourhood": (
        "MATCH (p:Paper {paper_id: 'W2626778328'})-[:AUTHORED_BY]->(a:Author)"
        "-[:WORKS_AT]->(i:Institution) RETURN p, a, i"
    ),
    "topic_to_institutions": (
        "MATCH (t:Topic {name: 'Natural Language Processing Techniques'})"
        "<-[:STUDIES]-(p:Paper)-[:AUTHORED_BY]->(:Author)-[:WORKS_AT]->(i:Institution) "
        "RETURN t, i, count(DISTINCT p) AS papers ORDER BY papers DESC LIMIT 15"
    ),
    "citation_subgraph": (
        "MATCH path = (p:Paper)-[:CITES]->(q:Paper) "
        "WITH p, count(*) AS outdeg ORDER BY outdeg DESC LIMIT 1 "
        "MATCH path = (p)-[:CITES]->(q:Paper) RETURN path"
    ),
}


def expected_counts(records: list[dict]) -> dict:
    """Distinct node/edge cardinalities the graph should contain (MERGE semantics)."""
    papers = {r["paper_id"] for r in records}
    authors, insts, topics = set(), set(), set()
    authored, works_at, studies, cites = set(), set(), set(), set()

    for r in records:
        pid = r["paper_id"]
        for a in r.get("authors") or []:
            aid = a["author_id"]
            authors.add(aid)
            authored.add((pid, aid))
            for inst in a.get("institutions") or []:
                iid = inst["inst_id"]
                insts.add(iid)
                works_at.add((aid, iid))
        for t in r.get("topics") or []:
            tid = t["topic_id"]
            topics.add(tid)
            studies.add((pid, tid))
        for ref in r.get("references") or []:
            if ref in papers:  # in-corpus only (builder MATCHes the destination)
                cites.add((pid, ref))

    return {
        "Paper": len(papers),
        "Author": len(authors),
        "Institution": len(insts),
        "Topic": len(topics),
        "AUTHORED_BY": len(authored),
        "WORKS_AT": len(works_at),
        "STUDIES": len(studies),
        "CITES": len(cites),
    }


def graph_counts(session) -> dict:
    out = {}
    for label in ("Paper", "Author", "Institution", "Topic"):
        out[label] = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
    for rtype in ("AUTHORED_BY", "WORKS_AT", "STUDIES", "CITES"):
        out[rtype] = session.run(
            f"MATCH ()-[r:{rtype}]->() RETURN count(r) AS c"
        ).single()["c"]
    return out


def integrity_checks(session) -> list[tuple[str, int, bool]]:
    """Return (label, count, ok) rows; ok=True means the count is acceptable."""
    checks = []

    def scalar(q):
        return session.run(q).single()["c"]

    # Duplicate ids should be impossible (uniqueness constraints) -> expect 0.
    dup_papers = scalar(
        "MATCH (p:Paper) WITH p.paper_id AS id, count(*) AS c WHERE c > 1 "
        "RETURN count(*) AS c"
    )
    checks.append(("duplicate paper ids", dup_papers, dup_papers == 0))

    # Self-citations (p CITES p) should not exist.
    self_cites = scalar("MATCH (p:Paper)-[:CITES]->(p) RETURN count(*) AS c")
    checks.append(("self-citations", self_cites, self_cites == 0))

    # Stray empty Paper nodes (no title) -> CITES MATCH should prevent these.
    empty_papers = scalar(
        "MATCH (p:Paper) WHERE p.title IS NULL RETURN count(*) AS c"
    )
    checks.append(("papers without a title", empty_papers, empty_papers == 0))

    # Informational (not failures): coverage gaps inherited from the data.
    no_author = scalar(
        "MATCH (p:Paper) WHERE NOT (p)-[:AUTHORED_BY]->(:Author) RETURN count(*) AS c"
    )
    checks.append(("papers with no author (info)", no_author, True))
    no_topic = scalar(
        "MATCH (p:Paper) WHERE NOT (p)-[:STUDIES]->(:Topic) RETURN count(*) AS c"
    )
    checks.append(("papers with no topic (info)", no_topic, True))

    return checks


def spot_check_traversals(session) -> list[str]:
    """Run a few multi-hop reads and render them as text lines."""
    lines = []

    lines.append("Top 5 institutions by paper count (Inst<-WorksAt-Author<-Authored-Paper):")
    q = (
        "MATCH (i:Institution)<-[:WORKS_AT]-(:Author)<-[:AUTHORED_BY]-(p:Paper) "
        "RETURN i.name AS inst, count(DISTINCT p) AS papers "
        "ORDER BY papers DESC LIMIT 5"
    )
    for r in session.run(q):
        lines.append(f"    {r['papers']:>5}  {r['inst']}")

    lines.append("Top 5 most-cited (in-corpus) papers by CITES in-degree:")
    q = (
        "MATCH (p:Paper)<-[:CITES]-() "
        "RETURN p.title AS title, count(*) AS indeg ORDER BY indeg DESC LIMIT 5"
    )
    for r in session.run(q):
        lines.append(f"    {r['indeg']:>5}  {r['title'][:70]}")

    lines.append(
        "3-hop sample — authors at Google studying "
        "'Natural Language Processing Techniques':"
    )
    q = (
        "MATCH (i:Institution {name: 'Google'})<-[:WORKS_AT]-(a:Author)"
        "<-[:AUTHORED_BY]-(p:Paper)-[:STUDIES]->(t:Topic "
        "{name: 'Natural Language Processing Techniques'}) "
        "RETURN count(DISTINCT a) AS authors, count(DISTINCT p) AS papers"
    )
    r = session.run(q).single()
    lines.append(f"    authors={r['authors']}  papers={r['papers']}")

    return lines


def main() -> int:
    in_path = os.path.join(config.DATA_INTERIM, "papers_normalized.jsonl")
    if not os.path.exists(in_path):
        print(f"missing {in_path} -- run src/ingest/normalize.py first", file=sys.stderr)
        return 1
    with open(in_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    expected = expected_counts(records)

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    try:
        with driver.session() as session:
            actual = graph_counts(session)
            checks = integrity_checks(session)
            traversals = spot_check_traversals(session)
    finally:
        driver.close()

    all_ok = True
    print("Counts (expected from JSONL vs. graph):")
    print(f"  {'entity':<14} {'expected':>10} {'graph':>10}   status")
    for key in expected:
        exp, act = expected[key], actual[key]
        ok = exp == act
        all_ok = all_ok and ok
        print(f"  {key:<14} {exp:>10,} {act:>10,}   {'OK' if ok else 'MISMATCH'}")

    print("\nIntegrity checks:")
    for label, count, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'ok' if ok else '!!'}] {label}: {count:,}")

    print("\nSpot-check traversals:")
    for line in traversals:
        print("  " + line)

    print("\nCypher for Neo4j Browser (http://localhost:7474) — M4.4 screenshots:")
    for name, q in SAMPLE_QUERIES.items():
        print(f"  -- {name}\n  {q}\n")

    print("RESULT:", "ALL OK" if all_ok else "FAILURES (see above)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
