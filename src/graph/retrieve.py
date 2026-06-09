"""Map a natural-language question to a graph traversal (M6.2).

`graph_retrieve(query)` is the graph half of the discovery engine: it picks one
of the M6.1 templates, resolves the entity the question is about against the live
graph, runs the traversal, and returns **structured context** (the rows + a
rendered text block) ready for the LLM answer-writer (M8) and the router (M7).

The mapping is deterministic and fully local — no LLM, no API:

  1. **Resolve the seed entity.** Author/Institution names are matched by
     "does the question contain this node's name?" (case-insensitive, length-
     guarded; authors must be a multi-word name). Topics — whose names are long
     descriptive phrases unlikely to appear verbatim — are matched by token
     overlap against the question. First match wins, in the priority
     author > institution > topic (most specific entity first).
  2. **Pick the template** from the seed type + intent cues in the question
     ("collaborate", "where … work", "same topics", "cite", "topics", …).
  3. **Run it** via `templates.run_template` and render the rows.

If no known entity is recognized the retriever abstains (`template=None`), which
is the signal M7 uses to fall back to vector RAG.

    python src/graph/retrieve.py "Which institutions work on the same topics as Ashish Vaswani?"
    python src/graph/retrieve.py "Who collaborates with Google?"
    python src/graph/retrieve.py --k 10 "What topics does Tsinghua University work on?"

Used by: hybrid router (M7), evaluation (M9).
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from graph import templates  # noqa: E402

DEFAULT_LIMIT = 15

# Min name length to consider for a substring seed match (avoids "Li" matching
# any word containing "li"). Authors additionally must be multi-word.
_MIN_AUTHOR_LEN = 6
_MIN_INST_LEN = 4
# Fraction of a topic's name tokens that must appear in the question to match.
_TOPIC_MATCH_THRESHOLD = 0.6

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an of on in for to with at by from about as and or".split()
)

# Intent cues (substring match against the lowercased question). Order of the
# checks in `select_template` encodes precedence, not the order here.
_COLLAB = (
    "collaborat", "co-author", "coauthor", "co author",
    "work with", "works with", "working with", "partner", "co-publish", "copublish",
)
_PEER = ("same topic", "similar", "peer", "same area", "same field", "also work",
         "else work", "else research", "same research")
_CITE = ("cite", "citing", "build on", "builds on", "built on", "based on",
         "influenc", "reference")
_TOPIC = ("topic", "research", "work on", "works on", "study", "studies",
          "studying", "area", "focus", "field", "subject", "specializ")
_INST = ("institution", "universit", "organization", "organisation", "org ",
         "lab", "laborator", "compan", "affiliat", "where", "work at", "based")
_PAPER = ("paper", "publication", "article", "influential", "seminal", "cited",
          "important")


def _tokens(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS
    }


def _has(query: str, cues: tuple[str, ...]) -> bool:
    q = query.lower()
    return any(c in q for c in cues)


# --- template selection (pure) ------------------------------------------------

def select_template(seed_type: str, query: str) -> str | None:
    """Pick a template name from the resolved seed type + the question's cues.

    Precedence within each seed type matters: more specific intents
    (cite/peer/collaborate) are checked before the generic fallback.
    """
    if seed_type == "author":
        if _has(query, _CITE):
            return "author_cited_institutions"
        if _has(query, _PEER):
            return "peer_institutions_via_topics"
        if _has(query, _COLLAB):
            return "coauthors"
        if _has(query, _INST):
            return "author_institutions"
        return "author_topics"
    if seed_type == "institution":
        if _has(query, _COLLAB):
            return "collaborating_institutions"
        if _has(query, _TOPIC):
            return "institution_topics"
        return "institution_authors"
    if seed_type == "topic":
        if _has(query, _PAPER):
            return "topic_top_cited_papers"
        if _has(query, _INST):
            return "topic_institutions"
        return "topic_authors"
    return None


# --- entity resolution --------------------------------------------------------

def _best_topic(query: str, names: list[str]) -> str | None:
    """Topic whose name has the highest token overlap with the question."""
    qtokens = _tokens(query)
    best, best_score = None, 0.0
    for name in names:
        ntokens = _tokens(name)
        if not ntokens:
            continue
        score = len(qtokens & ntokens) / len(ntokens)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= _TOPIC_MATCH_THRESHOLD else None


def resolve_authors(session, query: str, k: int = 3) -> list[str]:
    """Author nodes whose (multi-word) name appears in the question."""
    rows = session.run(
        "MATCH (a:Author) "
        "WHERE a.name CONTAINS ' ' AND size(a.name) >= $minlen "
        "AND toLower($q) CONTAINS toLower(a.name) "
        "RETURN a.name AS name ORDER BY size(a.name) DESC LIMIT $k",
        q=query, minlen=_MIN_AUTHOR_LEN, k=k,
    )
    return [r["name"] for r in rows]


def resolve_institutions(session, query: str, k: int = 3) -> list[str]:
    """Institution nodes whose name appears in the question."""
    rows = session.run(
        "MATCH (i:Institution) "
        "WHERE size(i.name) >= $minlen AND toLower($q) CONTAINS toLower(i.name) "
        "RETURN i.name AS name ORDER BY size(i.name) DESC LIMIT $k",
        q=query, minlen=_MIN_INST_LEN, k=k,
    )
    return [r["name"] for r in rows]


def resolve_topic(session, query: str) -> str | None:
    """Topic whose name best overlaps the question (token overlap, threshold)."""
    names = [r["name"] for r in session.run("MATCH (t:Topic) RETURN t.name AS name")]
    return _best_topic(query, names)


def resolve_seed(session, query: str) -> tuple[str | None, str | None]:
    """Resolve the seed entity: (seed_type, seed_name), most specific first."""
    authors = resolve_authors(session, query)
    if authors:
        return "author", authors[0]
    insts = resolve_institutions(session, query)
    if insts:
        return "institution", insts[0]
    topic = resolve_topic(session, query)
    if topic:
        return "topic", topic
    return None, None


# Seed type -> the template param name that carries the seed.
_PARAM_FOR_SEED = {
    "author": "author_name",
    "institution": "inst_name",
    "topic": "topic_name",
}


def render_context(template_name: str, seed_type: str, seed_name: str, rows: list[dict]) -> str:
    """Render the traversal result as a labelled text block for the LLM (M8)."""
    tmpl = templates.TEMPLATES[template_name]
    head = (
        f"Graph traversal `{template_name}` "
        f"(seed {seed_type} = \"{seed_name}\")\n{tmpl.description}"
    )
    if not rows:
        return head + "\n  (no matching paths in the graph)"
    lines = []
    for i, row in enumerate(rows, 1):
        parts = ", ".join(f"{k}={v}" for k, v in row.items())
        lines.append(f"  {i}. {parts}")
    return head + "\n" + "\n".join(lines)


def graph_retrieve(query: str, session=None, limit: int = DEFAULT_LIMIT) -> dict:
    """Map `query` to a graph traversal and return structured context.

    Returns a dict with: query, seed_type, seed_name, template, params, rows,
    context (rendered text). When no entity/template is found, `template` is
    None and `context` explains the abstention (router fallback signal).
    """
    own_driver = None
    if session is None:
        own_driver = _driver()
        session = own_driver.session()
    try:
        seed_type, seed_name = resolve_seed(session, query)
        if seed_type is None:
            return _abstain(query, "no known author, institution, or topic recognized")

        template_name = select_template(seed_type, query)
        if template_name is None:  # defensive; select_template covers every seed type
            return _abstain(query, f"no template for seed type {seed_type!r}")

        param = _PARAM_FOR_SEED[seed_type]
        rows = templates.run_template(session, template_name, {param: seed_name}, limit=limit)
        return {
            "query": query,
            "seed_type": seed_type,
            "seed_name": seed_name,
            "template": template_name,
            "params": {param: seed_name},
            "rows": rows,
            "context": render_context(template_name, seed_type, seed_name, rows),
        }
    finally:
        if own_driver is not None:
            session.close()
            own_driver.close()


def _abstain(query: str, reason: str) -> dict:
    return {
        "query": query,
        "seed_type": None,
        "seed_name": None,
        "template": None,
        "params": {},
        "rows": [],
        "context": f"No graph traversal: {reason}.",
    }


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def main() -> int:
    args = sys.argv[1:]
    limit = DEFAULT_LIMIT
    if "--k" in args:
        i = args.index("--k")
        limit = int(args[i + 1])
        del args[i : i + 2]
    if not args:
        print('usage: python src/graph/retrieve.py [--k N] "your question"', file=sys.stderr)
        return 1
    query = " ".join(args)

    result = graph_retrieve(query, limit=limit)
    print(f"Q: {result['query']}")
    if result["template"] is None:
        print(result["context"])
        return 0
    print(
        f"seed: {result['seed_type']} = {result['seed_name']!r}  "
        f"-> template: {result['template']}  ({len(result['rows'])} rows)\n"
    )
    print(result["context"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
