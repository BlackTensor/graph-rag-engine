"""End-to-end `answer(query)` through the router (M7.3).

The single entry point for the whole discovery engine. It threads a question
through everything built in M5–M7 and returns a grounded answer plus the full
provenance the demo (M10) and eval (M9) need:

    answer(query)
      └─ router.route(query)          # M7.1 classify -> M7.x retrieve -> M7.2 merge
           ├─ classify  -> vector | graph | hybrid (+ graph-abstain fallback)
           ├─ retrieve  -> graph_retrieve / search
           └─ merge     -> one deduped, labelled context block
      └─ ground + generate            # answer ONLY from the merged context
      └─ {query, answer, route, reason, context, passages, graph_result, ...}

The LLM call here is the same minimal `rag.generate()` the baselines use — M8
replaces it with a hardened, reusable Ollama wrapper and makes it *the* writer
for every pipeline (M8.2). What M7.3 locks in is the **wiring**: any question,
routed and answered through one call, grounded on the merged graph+vector
context (never outside knowledge).

    python src/retrieval/pipeline.py "Which institutions collaborate most with Tsinghua University?"
    python src/retrieval/pipeline.py "What is the Transformer architecture?"

Used by: Streamlit demo (M10), evaluation harness (M9).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval import router  # noqa: E402
from vector import rag  # noqa: E402

# Unified answer-writer prompt: the merged context may carry knowledge-graph
# facts (precise relationships/rankings) and/or document passages (title+abstract).
SYSTEM_PROMPT = (
    "You are a research assistant for an AI-research-papers knowledge base. Answer "
    "the question using ONLY the context below. The context may contain "
    "knowledge-graph facts (exact relationships, affiliations, citations, and "
    "ranked counts) and/or document passages (a paper's title and abstract). For "
    "questions about who works with or at whom, who cites whom, or which "
    "entities rank highest, rely on the graph facts and report the ranked items "
    "with their counts. If the context does not contain the answer, say you don't "
    "know — do not use outside knowledge. Be concise."
)


def build_prompt(query: str, context: str) -> str:
    """Assemble the grounded prompt from the router's merged context block."""
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def answer(query: str, model: str | None = None, host: str | None = None) -> dict:
    """Route `query`, then write a grounded answer from the merged context.

    Returns the answer plus provenance: which route ran and why, what was
    executed (incl. any graph->vector fallback), the merged context the answer
    was grounded on, the deduped passages, and the raw graph traversal result.
    """
    state = router.route(query)
    context = state.get("merged_context", "(no context retrieved)")
    prompt = build_prompt(query, context)
    text = rag.generate(prompt, model=model, host=host, system=SYSTEM_PROMPT)
    return {
        "query": query,
        "answer": text,
        "route": state.get("route"),
        "reason": state.get("reason"),
        "executed": state.get("executed", []),
        "fellback": state.get("fellback", False),
        "context": context,
        "passages": state.get("passages", []),
        "graph_result": state.get("graph_result"),
    }


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print('usage: python src/retrieval/pipeline.py "your question"', file=sys.stderr)
        return 1
    query = " ".join(args)

    result = answer(query)
    print(f"Q: {result['query']}")
    print(f"route: {result['route']}  ({result['reason']})")
    print(f"executed: {', '.join(result['executed'] or ['(none)'])}")
    if result["fellback"]:
        print("note: graph abstained -> fell back to vector")
    print(f"\nA: {result['answer']}\n")
    print("--- grounded on ---")
    print(result["context"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
