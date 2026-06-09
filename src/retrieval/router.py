"""LangGraph router: send each question to the right retriever (M7.1).

The discovery engine has two retrievers — vector RAG (M5, good at semantic /
definitional questions over title+abstract text) and Graph-RAG (M6, good at
relational / multi-hop questions over the knowledge graph). This module decides
which one a question needs and routes it through a small LangGraph state machine:

        ┌──────────┐   route == "vector"   ┌──────────┐
        │          │──────────────────────▶│  vector  │
   ▶───▶│ classify │   route == "graph"    ├──────────┤     ┌───────┐
        │          │──────────────────────▶│  graph   │────▶│ merge │──▶ END
        └──────────┘   route == "hybrid"   ├──────────┤     └───────┘
                                           │  hybrid  │
                                           └──────────┘

Two layers of decision, by design:

  1. **Lexical classification** (`classify`, no DB, no LLM — deterministic and
     instant): does the question carry *relational* signals (collaborate, cite,
     same-topics, who-works-where, "which institutions/authors …") → graph; is
     it purely *definitional / semantic* ("what is", "explain", "how does …") →
     vector; does it carry both → hybrid (run both, merge in M7.2).
  2. **Runtime hybrid fallback** (inside `graph` node): the lexical layer can't
     know whether a named entity actually exists in the graph. The graph node
     runs `graph_retrieve`, and if it *abstains* (no Author/Institution/Topic
     resolves), it falls back to vector retrieval so the question is still
     answered. This is the safety net that makes lenient graph routing safe.

After the chosen retriever(s) run, a `merge` node (M7.2, `retrieval/merge.py`)
combines graph + vector context into one deduped, prompt-ready block in
`merged_context`. Writing the final grounded answer from it is M7.3 — the nodes
here stop at retrieval + merge.

    python src/retrieval/router.py "Which institutions collaborate most with Tsinghua University?"
    python src/retrieval/router.py "What is the Transformer architecture?"

Used by: hybrid context-merge (M7.2), answer() pipeline (M7.3), eval (M9).
"""

from __future__ import annotations

import os
import string
import sys
from dataclasses import dataclass
from typing import Optional, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph import retrieve  # noqa: E402
from retrieval import merge  # noqa: E402
from vector import search  # noqa: E402

DEFAULT_K = 5

# --- classification cues ------------------------------------------------------
# All matched as substrings against the lowercased question. Tuned for routing
# (vector vs graph), not template selection (that's retrieve.select_template).

# Relational / multi-hop signals -> the answer is about how entities relate
# (who works with / at / cites whom, who shares topics), which lives only in the
# graph — never in an abstract chunk.
_RELATIONAL = (
    "collaborat", "co-author", "coauthor", "co author", "co-publish", "copublish",
    "work with", "works with", "working with", "partner",
    "cite", "citing", "cited by", "who cite", "whose work",
    "same topic", "same area", "same field", "same research", "similar work",
    "peer institution", "also work on", "else work on",
    "who works at", "who work at", "where does", "where do", "works at", "work at",
    "based at", "affiliat", "who publishes", "who publish",
)

# "which/who <entity-type>" — asking to enumerate graph entities (institutions,
# authors, …). Vector RAG can't enumerate these; the graph can.
_ENTITY_TYPE = (
    "institution", "universit", "organi", "compan", "laborator", " lab",
    "researcher", "scientist", "collaborator", "co-author", "coauthor",
)

# Topic/affiliation intent that becomes a graph question when tied to a named
# entity (a mid-sentence proper noun, see `_has_proper_noun`).
_ENTITY_INTENT = (
    "research", "work on", "works on", "working on", "study", "studies",
    "studying", "topics", "focus on", "specializ", "publishes on", "publish on",
)

# Definitional / explanatory / semantic-content signals -> vector RAG.
_CONTENT = (
    "what is", "what are", "what's", "whats", "explain", "describe", "definition",
    "define", "how does", "how do", "how is", "how are", "summar", "tell me about",
    "overview of", "introduction to",
)

_QUESTION_WORDS = frozenset(
    "which who what where when how why whose does do is are can list name".split()
)


def _first_match(q: str, cues: tuple[str, ...]) -> Optional[str]:
    return next((c for c in cues if c in q), None)


def _which_entity(q: str) -> bool:
    """True for 'which/who/list … <entity-type>' enumeration questions."""
    if "which" in q or "who " in q or "list " in q or "name the" in q:
        return any(e in q for e in _ENTITY_TYPE)
    return False


def _has_proper_noun(query: str) -> bool:
    """A capitalized, mid-sentence, alphabetic token — a likely named entity.

    Skips the first word (questions open with a capitalized 'Which/What/…') and
    question words, so 'What does Google research?' flags Google but 'What is
    attention?' flags nothing.
    """
    for w in query.split()[1:]:
        cw = w.strip(string.punctuation)
        if len(cw) >= 2 and cw[0].isupper() and cw.isalpha() and cw.lower() not in _QUESTION_WORDS:
            return True
    return False


@dataclass(frozen=True)
class RouteDecision:
    """The classifier's verdict: which retriever(s) and a human-readable why."""

    route: str  # "vector" | "graph" | "hybrid"
    reason: str


def classify(query: str) -> RouteDecision:
    """Classify a question as vector / graph / hybrid (lexical, deterministic).

    graph  — carries a relational signal, or enumerates graph entities, or pairs
             topic/affiliation intent with a named entity.
    hybrid — carries BOTH a relational and a definitional/semantic signal.
    vector — everything else (the safe default: semantic text retrieval).
    """
    q = query.lower()
    rel = _first_match(q, _RELATIONAL)
    which = _which_entity(q)
    intent = _first_match(q, _ENTITY_INTENT)
    proper = _has_proper_noun(query)
    content = _first_match(q, _CONTENT)

    graphish = bool(rel) or which or (intent is not None and proper)
    if graphish:
        if rel:
            graph_why = f"relational cue '{rel}'"
        elif which:
            graph_why = "enumerates graph entities ('which/who <entity>')"
        else:
            graph_why = f"entity intent '{intent}' + named entity"
        if content:
            return RouteDecision(
                "hybrid", f"{graph_why} + content cue '{content}' — needs both retrievers"
            )
        return RouteDecision("graph", graph_why)

    if content:
        return RouteDecision("vector", f"definitional/semantic question ('{content}')")
    return RouteDecision("vector", "no relational signal; defaulting to semantic retrieval")


# --- LangGraph state machine --------------------------------------------------


class RouterState(TypedDict, total=False):
    """Channels threaded through the router graph.

    query           — the input question (set by the caller).
    route / reason  — the classifier's decision (set by `classify_node`).
    executed        — which retrievers actually ran (incl. any fallback).
    graph_result    — full graph_retrieve(...) output, or None.
    vector_contexts — search(...) hits, or None.
    fellback        — True if the graph node abstained and used vector instead.
    merged_context  — combined, deduped, prompt-ready context (set by `merge`).
    passages        — deduped vector passages (one per paper) behind the context.
    dropped         — raw vector chunks removed by dedupe.
    """

    query: str
    route: str
    reason: str
    executed: list[str]
    graph_result: Optional[dict]
    vector_contexts: Optional[list]
    fellback: bool
    merged_context: str
    passages: list[dict]
    dropped: int


def classify_node(state: RouterState) -> dict:
    decision = classify(state["query"])
    return {
        "route": decision.route,
        "reason": decision.reason,
        "executed": [],
        "fellback": False,
    }


def vector_node(state: RouterState) -> dict:
    contexts = search.search(state["query"], k=DEFAULT_K)
    return {"vector_contexts": contexts, "executed": ["vector"]}


def graph_node(state: RouterState) -> dict:
    """Run graph retrieval; fall back to vector when no entity resolves."""
    result = retrieve.graph_retrieve(state["query"])
    if result["template"] is None:
        # Lexically relational but no graph entity matched — hybrid fallback.
        contexts = search.search(state["query"], k=DEFAULT_K)
        return {
            "graph_result": result,
            "vector_contexts": contexts,
            "executed": ["graph(abstained)", "vector(fallback)"],
            "fellback": True,
        }
    return {"graph_result": result, "executed": ["graph"]}


def hybrid_node(state: RouterState) -> dict:
    """Run both retrievers; M7.2 merges their context."""
    result = retrieve.graph_retrieve(state["query"])
    contexts = search.search(state["query"], k=DEFAULT_K)
    return {
        "graph_result": result,
        "vector_contexts": contexts,
        "executed": ["graph", "vector"],
        "fellback": False,
    }


def merge_node(state: RouterState) -> dict:
    """Combine graph + vector context into one deduped, prompt-ready block (M7.2)."""
    merged = merge.merge_context(state.get("graph_result"), state.get("vector_contexts"))
    return {
        "merged_context": merged.text,
        "passages": merged.passages,
        "dropped": merged.dropped,
    }


def _select_route(state: RouterState) -> str:
    return state["route"]


def build_router():
    """Compile the LangGraph router (classify -> vector | graph | hybrid)."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(RouterState)
    graph.add_node("classify", classify_node)
    graph.add_node("vector", vector_node)
    graph.add_node("graph", graph_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("merge", merge_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _select_route,
        {"vector": "vector", "graph": "graph", "hybrid": "hybrid"},
    )
    # Every retrieval branch funnels through the merge step before finishing.
    graph.add_edge("vector", "merge")
    graph.add_edge("graph", "merge")
    graph.add_edge("hybrid", "merge")
    graph.add_edge("merge", END)
    return graph.compile()


_router = None


def route(query: str) -> RouterState:
    """Classify `query`, run the chosen retriever(s), return the final state.

    The compiled router is cached across calls (one build per process).
    """
    global _router
    if _router is None:
        _router = build_router()
    return _router.invoke({"query": query})


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print('usage: python src/retrieval/router.py "your question"', file=sys.stderr)
        return 1
    query = " ".join(args)

    # Show the routing decision without touching the DB/Qdrant first.
    decision = classify(query)
    print(f"Q: {query}")
    print(f"route: {decision.route}  ({decision.reason})\n")

    state = route(query)
    print(f"executed: {', '.join(state.get('executed') or ['(none)'])}")
    if state.get("fellback"):
        print("note: graph abstained -> fell back to vector")
    if state.get("dropped"):
        print(f"deduped: dropped {state['dropped']} redundant chunk(s)")

    print("\n--- merged context ---")
    print(state.get("merged_context", "(none)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
