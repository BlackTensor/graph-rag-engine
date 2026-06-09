"""Graph-RAG vs vector-RAG on a multi-hop question (M6.3 — the money shot).

Proves the North Star claim: a multi-hop question where the vector baseline
*fails* and Graph-RAG *nails it*, side by side.

Why vector RAG structurally can't answer the demo question
----------------------------------------------------------
The Qdrant index holds chunks of **title + abstract only** (M5.1) — author and
institution affiliations were never embedded, they live exclusively in the
graph. So any question about *who* works with *whom*, or which institutions
publish on a topic, is unanswerable from the retrieved passages no matter how
good the embedding is: the answer simply isn't in the text. Graph-RAG walks
`AUTHORED_BY`/`WORKS_AT`/`STUDIES` edges and returns the exact ranked answer.

`graph_rag(query)` mirrors `vector_rag(query)` (M5.3): retrieve -> ground ->
answer, but the retriever is `graph_retrieve` (M6.2) and the context is the
structured traversal result. The LLM call is the same minimal `generate()` the
baseline uses; M7 (router) + M8 (unified writer) replace this glue later.

    python src/graph/compare.py                       # run the default demo question
    python src/graph/compare.py "Who collaborates with Google?"

Uses: graph.retrieve (M6.2), vector.rag (M5.3).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph import retrieve  # noqa: E402
from vector import rag  # noqa: E402

# A genuinely multi-hop, relational question. The answer is an aggregation over
# co-authorship across institutions — nothing a single abstract chunk contains.
DEMO_QUESTION = "Which institutions collaborate most with Tsinghua University?"

GRAPH_SYSTEM_PROMPT = (
    "You are a research-graph assistant. Answer the question using ONLY the "
    "structured graph results below (each line is a real path found in the "
    "knowledge graph). If the results are empty, say you don't know. Be concise "
    "and report the ranked entities with their counts."
)


def build_graph_prompt(query: str, context: str) -> str:
    """Assemble the grounded prompt from a graph_retrieve `context` block."""
    return f"Graph results:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def graph_rag(
    query: str,
    session=None,
    limit: int = 15,
    model: str | None = None,
    host: str | None = None,
) -> dict:
    """Graph-RAG: graph_retrieve -> ground -> answer (mirrors vector_rag).

    Returns the full `graph_retrieve` result plus the LLM `answer`.
    """
    result = retrieve.graph_retrieve(query, session=session, limit=limit)
    prompt = build_graph_prompt(query, result["context"])
    result["answer"] = rag.generate(
        prompt, model=model, host=host, system=GRAPH_SYSTEM_PROMPT
    )
    return result


def compare(query: str, k: int = 5) -> dict:
    """Run both pipelines on `query`; return {vector, graph} result dicts."""
    return {"vector": rag.vector_rag(query, k=k), "graph": graph_rag(query)}


def main() -> int:
    args = sys.argv[1:]
    query = " ".join(args) if args else DEMO_QUESTION

    print(f"Question (multi-hop): {query}\n")
    print("=" * 72)
    print("VECTOR RAG (baseline — title+abstract chunks, no affiliation data)")
    print("=" * 72)
    v = rag.vector_rag(query, k=5)
    print(f"Answer: {v['answer']}\n")
    print("Retrieved passages:")
    for rank, c in enumerate(v["contexts"], 1):
        print(f"  [{rank}] ({c['score']:.4f}) {c.get('title', '')}")

    print("\n" + "=" * 72)
    print("GRAPH-RAG (traversal over AUTHORED_BY / WORKS_AT / STUDIES)")
    print("=" * 72)
    g = graph_rag(query)
    if g["template"] is None:
        print(f"(no traversal — {g['context']})")
        return 0
    print(
        f"seed: {g['seed_type']} = {g['seed_name']!r}  -> template: {g['template']}\n"
    )
    print(f"Answer: {g['answer']}\n")
    print("Traversal rows:")
    for i, row in enumerate(g["rows"], 1):
        print("  " + f"{i}. " + ", ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
