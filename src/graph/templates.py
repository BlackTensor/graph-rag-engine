"""Parametric Cypher templates for common multi-hop patterns (M6.1).

A curated library of *named, parameterized* traversals over the M4 graph
(`Paper`/`Author`/`Institution`/`Topic` + `AUTHORED_BY`/`WORKS_AT`/`STUDIES`/
`CITES`). Each template is a reusable Cypher string with `$placeholders` and a
short natural-language description — the building blocks the M6.2 retriever maps
a question onto, and the answers to multi-hop questions vector RAG can't reach
(M6.3).

Design notes
------------
* **Templates are precise; resolution is the caller's job.** Author/Institution
  seeds match on case-insensitive *equality* (`toLower(n.name) = toLower($x)`)
  so a seed picks out exactly one entity; Topic seeds match case-insensitive
  *substring* because topic names are long descriptive phrases
  ("Natural Language Processing Techniques"). Fuzzy NL→entity resolution lives
  in M6.2 — these templates assume a resolved name is handed in.
* **Every template ends in `LIMIT $limit`** so results stay bounded; the runner
  injects a default when the caller doesn't pass one.
* **`DISTINCT` + `count(DISTINCT …)`** everywhere a traversal can fan out, so a
  paper authored by five people at one institution doesn't inflate counts.

    python src/graph/templates.py --list
    python src/graph/templates.py author_topics author_name="Yoshua Bengio"
    python src/graph/templates.py peer_institutions_via_topics author_name="Ashish Vaswani" limit=10
    python src/graph/templates.py topic_institutions topic_name="Natural Language Processing"

The headline pattern is `peer_institutions_via_topics` — the literal
author→paper→topic→institution chain from the milestone: institutions publishing
on the same topics a given author studies.

Used by: graph_retrieve (M6.2), hybrid router (M7).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

DEFAULT_LIMIT = 15

# $placeholders the runner supplies itself; never required from the caller.
_RESERVED_PARAMS = frozenset({"limit"})
_PARAM_RE = re.compile(r"\$(\w+)")


@dataclass(frozen=True)
class QueryTemplate:
    """One named, parameterized traversal.

    `params` are the placeholders the caller MUST supply (everything in the
    Cypher except the reserved `$limit`). `examples` are sample NL questions the
    template answers — handy for the M6.2 router and for documentation.
    """

    name: str
    description: str
    params: tuple[str, ...]
    cypher: str
    examples: tuple[str, ...] = field(default_factory=tuple)

    def placeholders(self) -> set[str]:
        """Every distinct `$name` referenced in the Cypher."""
        return set(_PARAM_RE.findall(self.cypher))


def _t(name, description, params, cypher, examples=()):
    return QueryTemplate(name, description, tuple(params), cypher.strip(), tuple(examples))


# --- The template library -----------------------------------------------------
# Grouped by seed entity. Keep this list curated: each template earns its place
# by answering a distinct relational question.

_TEMPLATES: list[QueryTemplate] = [
    # --- seed: Author ---------------------------------------------------------
    _t(
        "author_topics",
        "Topics a given author studies, by paper count "
        "(Author<-AUTHORED_BY-Paper-STUDIES->Topic).",
        ["author_name"],
        """
        MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:STUDIES]->(t:Topic)
        WHERE toLower(a.name) = toLower($author_name)
        RETURN t.name AS topic, count(DISTINCT p) AS papers
        ORDER BY papers DESC, topic
        LIMIT $limit
        """,
        ["What topics does Yoshua Bengio work on?", "What does <author> research?"],
    ),
    _t(
        "author_institutions",
        "Institutions a given author is affiliated with "
        "(Author-WORKS_AT->Institution).",
        ["author_name"],
        """
        MATCH (a:Author)-[:WORKS_AT]->(i:Institution)
        WHERE toLower(a.name) = toLower($author_name)
        RETURN DISTINCT i.name AS institution
        ORDER BY institution
        LIMIT $limit
        """,
        ["Where does <author> work?", "Which institution is <author> affiliated with?"],
    ),
    _t(
        "coauthors",
        "An author's collaborators, by number of shared papers "
        "(Author<-AUTHORED_BY-Paper-AUTHORED_BY->Author).",
        ["author_name"],
        """
        MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(co:Author)
        WHERE toLower(a.name) = toLower($author_name) AND co <> a
        RETURN co.name AS coauthor, count(DISTINCT p) AS shared_papers
        ORDER BY shared_papers DESC, coauthor
        LIMIT $limit
        """,
        ["Who collaborates with <author>?", "Who are <author>'s co-authors?"],
    ),
    _t(
        "peer_institutions_via_topics",
        "Institutions publishing on the same topics a given author studies — the "
        "author->paper->topic->institution chain "
        "(Author<-AUTHORED_BY-Paper-STUDIES->Topic<-STUDIES-Paper-AUTHORED_BY->"
        "Author-WORKS_AT->Institution).",
        ["author_name"],
        """
        MATCH (a:Author)<-[:AUTHORED_BY]-(:Paper)-[:STUDIES]->(t:Topic)
        WHERE toLower(a.name) = toLower($author_name)
        WITH collect(DISTINCT t) AS topics
        UNWIND topics AS t
        MATCH (t)<-[:STUDIES]-(p2:Paper)-[:AUTHORED_BY]->(:Author)-[:WORKS_AT]->(i:Institution)
        RETURN i.name AS institution,
               count(DISTINCT p2) AS papers,
               count(DISTINCT t)  AS shared_topics
        ORDER BY papers DESC, institution
        LIMIT $limit
        """,
        [
            "Which institutions work on the same topics as <author>?",
            "Who else researches what <author> researches?",
        ],
    ),
    _t(
        "author_cited_institutions",
        "Institutions whose work an author cites — citation reach across orgs "
        "(Author<-AUTHORED_BY-Paper-CITES->Paper-AUTHORED_BY->Author-WORKS_AT->"
        "Institution). Sparse: CITES is in-corpus only (§10).",
        ["author_name"],
        """
        MATCH (a:Author)<-[:AUTHORED_BY]-(:Paper)-[:CITES]->(cited:Paper)
        WHERE toLower(a.name) = toLower($author_name)
        MATCH (cited)-[:AUTHORED_BY]->(:Author)-[:WORKS_AT]->(i:Institution)
        RETURN i.name AS institution, count(DISTINCT cited) AS cited_papers
        ORDER BY cited_papers DESC, institution
        LIMIT $limit
        """,
        ["Whose work does <author> build on?", "Which labs does <author> cite?"],
    ),
    # --- seed: Institution ----------------------------------------------------
    _t(
        "institution_authors",
        "Most-published authors affiliated with a given institution "
        "(Institution<-WORKS_AT-Author<-AUTHORED_BY-Paper).",
        ["inst_name"],
        """
        MATCH (i:Institution)<-[:WORKS_AT]-(a:Author)<-[:AUTHORED_BY]-(p:Paper)
        WHERE toLower(i.name) = toLower($inst_name)
        RETURN a.name AS author, count(DISTINCT p) AS papers
        ORDER BY papers DESC, author
        LIMIT $limit
        """,
        ["Who publishes at Google?", "Which authors are at <institution>?"],
    ),
    _t(
        "institution_topics",
        "Topics a given institution works on, by paper count "
        "(Institution<-WORKS_AT-Author<-AUTHORED_BY-Paper-STUDIES->Topic).",
        ["inst_name"],
        """
        MATCH (i:Institution)<-[:WORKS_AT]-(:Author)<-[:AUTHORED_BY]-(p:Paper)
              -[:STUDIES]->(t:Topic)
        WHERE toLower(i.name) = toLower($inst_name)
        RETURN t.name AS topic, count(DISTINCT p) AS papers
        ORDER BY papers DESC, topic
        LIMIT $limit
        """,
        ["What does Stanford work on?", "Which topics does <institution> research?"],
    ),
    _t(
        "collaborating_institutions",
        "Institutions that co-author papers with a given institution "
        "(Institution<-WORKS_AT-Author<-AUTHORED_BY-Paper-AUTHORED_BY->Author"
        "-WORKS_AT->Institution).",
        ["inst_name"],
        """
        MATCH (i:Institution)<-[:WORKS_AT]-(:Author)<-[:AUTHORED_BY]-(p:Paper)
              -[:AUTHORED_BY]->(:Author)-[:WORKS_AT]->(other:Institution)
        WHERE toLower(i.name) = toLower($inst_name) AND other <> i
        RETURN other.name AS institution, count(DISTINCT p) AS shared_papers
        ORDER BY shared_papers DESC, institution
        LIMIT $limit
        """,
        ["Who collaborates with DeepMind?", "Which orgs co-publish with <institution>?"],
    ),
    # --- seed: Topic ----------------------------------------------------------
    _t(
        "topic_institutions",
        "Institutions publishing most on a given topic "
        "(Topic<-STUDIES-Paper-AUTHORED_BY->Author-WORKS_AT->Institution).",
        ["topic_name"],
        """
        MATCH (t:Topic)<-[:STUDIES]-(p:Paper)-[:AUTHORED_BY]->(:Author)
              -[:WORKS_AT]->(i:Institution)
        WHERE toLower(t.name) CONTAINS toLower($topic_name)
        RETURN i.name AS institution, count(DISTINCT p) AS papers
        ORDER BY papers DESC, institution
        LIMIT $limit
        """,
        ["Which institutions lead on retrieval-augmented generation?"],
    ),
    _t(
        "topic_authors",
        "Most-published authors on a given topic "
        "(Topic<-STUDIES-Paper-AUTHORED_BY->Author).",
        ["topic_name"],
        """
        MATCH (t:Topic)<-[:STUDIES]-(p:Paper)-[:AUTHORED_BY]->(a:Author)
        WHERE toLower(t.name) CONTAINS toLower($topic_name)
        RETURN a.name AS author, count(DISTINCT p) AS papers
        ORDER BY papers DESC, author
        LIMIT $limit
        """,
        ["Who are the leading authors on multi-hop reasoning?"],
    ),
    _t(
        "topic_top_cited_papers",
        "Most in-corpus-cited papers on a given topic "
        "(Topic<-STUDIES-Paper<-CITES-Paper).",
        ["topic_name"],
        """
        MATCH (t:Topic)<-[:STUDIES]-(p:Paper)
        WHERE toLower(t.name) CONTAINS toLower($topic_name)
        OPTIONAL MATCH (p)<-[c:CITES]-(:Paper)
        RETURN p.title AS paper, p.year AS year, count(c) AS in_citations
        ORDER BY in_citations DESC, paper
        LIMIT $limit
        """,
        ["What are the most influential papers on knowledge graphs?"],
    ),
]

# Name -> template registry (definition order preserved by dict in 3.7+).
TEMPLATES: dict[str, QueryTemplate] = {t.name: t for t in _TEMPLATES}


def list_templates() -> list[QueryTemplate]:
    return list(TEMPLATES.values())


def run_template(
    session,
    name: str,
    params: dict | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """Run a named template against an open Neo4j session, return rows as dicts.

    Validates that every required placeholder is supplied (and that no unknown
    params are passed), then injects `$limit`.
    """
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r} (see list_templates())")
    template = TEMPLATES[name]
    params = dict(params or {})

    missing = [p for p in template.params if p not in params]
    if missing:
        raise ValueError(f"template {name!r} missing params: {', '.join(missing)}")
    unknown = [k for k in params if k not in template.params]
    if unknown:
        raise ValueError(f"template {name!r} got unknown params: {', '.join(unknown)}")

    result = session.run(template.cypher, **params, limit=limit)
    return [dict(record) for record in result]


def render_rows(rows: list[dict]) -> str:
    """Render result rows as a simple aligned table for the CLI."""
    if not rows:
        return "  (no results)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  " + "  ".join("-" * widths[c] for c in cols)
    body = [
        "  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows
    ]
    return "\n".join([header, sep, *body])


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def _print_list() -> None:
    print(f"{len(TEMPLATES)} templates:\n")
    for t in list_templates():
        plist = ", ".join(t.params) or "(none)"
        print(f"  {t.name}")
        print(f"      params: {plist}")
        print(f"      {t.description}")
        if t.examples:
            print(f"      e.g. {t.examples[0]}")
        print()


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("--list", "-l", "--help", "-h"):
        _print_list()
        return 0

    name = args[0]
    if name not in TEMPLATES:
        print(f"unknown template {name!r}; run --list to see them.", file=sys.stderr)
        return 2

    # Parse key=value pairs; `limit` is special.
    params: dict = {}
    limit = DEFAULT_LIMIT
    for tok in args[1:]:
        if "=" not in tok:
            print(f"bad arg {tok!r}; expected key=value", file=sys.stderr)
            return 2
        key, val = tok.split("=", 1)
        if key == "limit":
            limit = int(val)
        else:
            params[key] = val

    driver = _driver()
    try:
        with driver.session() as session:
            rows = run_template(session, name, params, limit=limit)
    finally:
        driver.close()

    print(f"Template: {name}  params={params}  limit={limit}\n")
    print(render_rows(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
