"""Merge graph + vector context into one clean, deduped block (M7.2).

The router (M7.1) can run either retriever or both (hybrid / graph-abstain
fallback). Before the answer-writer (M7.3 / M8) sees the context, the two
modalities are combined into a single prompt-ready block and deduplicated:

  * **Within vector results** — Qdrant returns *chunks*, and a long paper has
    several. Multiple chunks of the same `paper_id` are the same document, so we
    collapse to one passage per paper, keeping the highest-scoring chunk (results
    arrive score-descending, so that's the first one seen).
  * **Across modalities** — a paper named in the graph rows (e.g. a
    `topic_top_cited_papers` traversal) may also surface as a vector passage.
    When a passage's title matches a value already in the graph facts, it's
    dropped — the graph already surfaced it, no need to print it twice.

The output keeps the two modalities in **labelled sections** (graph facts first —
they're the precise relational answer — then supporting document passages) so the
LLM can tell a structured fact from a retrieved snippet. Pure-vector and
pure-graph routes naturally produce a single section.

This module is pure (no DB, no LLM): `merge_context(graph_result, vector_contexts)`
takes the two retrieval outputs straight from `RouterState` and returns a
`MergedContext`. Wired into the router as the `merge` node.

Used by: router merge node (M7.1/M7.2), answer() pipeline (M7.3), eval (M9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MergedContext:
    """The combined, deduped context plus the bookkeeping the demo/eval want.

    text         — prompt-ready block (labelled sections), or a no-context note.
    graph_facts  — the graph traversal's rendered context, or None.
    passages     — deduped vector passages, one per paper (highest-scoring chunk).
    n_graph_rows — number of graph rows behind `graph_facts`.
    n_passages   — len(passages).
    dropped      — raw vector chunks removed by dedupe (same paper / cross-modal).
    """

    text: str
    graph_facts: Optional[str]
    passages: list[dict]
    n_graph_rows: int
    n_passages: int
    dropped: int


def _graph_row_values(graph_result: dict) -> frozenset[str]:
    """Normalized string values appearing in the graph rows (for cross-dedupe)."""
    vals = set()
    for row in graph_result.get("rows", []):
        for v in row.values():
            if isinstance(v, str) and v.strip():
                vals.add(v.strip().lower())
    return frozenset(vals)


def dedupe_passages(
    contexts: list[dict], graph_values: frozenset[str] = frozenset()
) -> tuple[list[dict], int]:
    """Collapse vector chunks to one passage per paper; drop graph-named papers.

    Assumes `contexts` is score-descending (as `search` returns), so the first
    chunk seen for a paper is its best. Returns (passages, dropped_count).
    """
    seen: dict = {}
    order: list = []
    dropped = 0
    for c in contexts:
        title_norm = (c.get("title") or "").strip().lower()
        # Cross-modal: this paper is already named in the graph facts.
        if title_norm and title_norm in graph_values:
            dropped += 1
            continue
        key = c.get("paper_id") or c.get("chunk_id") or title_norm or id(c)
        if key in seen:
            dropped += 1
            continue
        seen[key] = c
        order.append(key)
    return [seen[k] for k in order], dropped


def _passage_body(c: dict) -> str:
    """The chunk text without a leading duplicate of the title (chunk 0 repeats it)."""
    text = (c.get("text") or "").strip()
    title = (c.get("title") or "").strip()
    if title and text.startswith(title):
        text = text[len(title):].lstrip("\n").strip()
    return text


def format_passages(passages: list[dict]) -> str:
    """Render deduped passages as numbered `[n] Title (year, N cites)` blocks."""
    blocks = []
    for i, c in enumerate(passages, 1):
        title = (c.get("title") or "").strip()
        meta = []
        if c.get("year"):
            meta.append(str(c["year"]))
        if c.get("cited_by_count") is not None:
            meta.append(f"{c['cited_by_count']} cites")
        header = f"[{i}] {title}" + (f" ({', '.join(meta)})" if meta else "")
        body = _passage_body(c)
        blocks.append(f"{header}\n{body}" if body else header)
    return "\n\n".join(blocks)


def merge_context(
    graph_result: Optional[dict], vector_contexts: Optional[list[dict]]
) -> MergedContext:
    """Combine graph + vector retrieval into one deduped, labelled context block."""
    graph_facts = None
    graph_values: frozenset[str] = frozenset()
    n_graph_rows = 0
    if graph_result and graph_result.get("template"):
        graph_facts = graph_result["context"]
        graph_values = _graph_row_values(graph_result)
        n_graph_rows = len(graph_result.get("rows", []))

    passages, dropped = dedupe_passages(vector_contexts or [], graph_values)

    sections = []
    if graph_facts:
        sections.append("Knowledge-graph facts:\n" + graph_facts)
    if passages:
        sections.append("Document passages:\n" + format_passages(passages))
    text = "\n\n".join(sections) if sections else "(no context retrieved)"

    return MergedContext(
        text=text,
        graph_facts=graph_facts,
        passages=passages,
        n_graph_rows=n_graph_rows,
        n_passages=len(passages),
        dropped=dropped,
    )
