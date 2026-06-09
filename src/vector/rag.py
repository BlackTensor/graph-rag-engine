"""Baseline vector-RAG pipeline: retrieve -> ground -> answer (M5.3, M8.2).

`vector_rag(query)` is the traditional-RAG baseline the Graph-RAG engine has to
beat on multi-hop questions (CLAUDE.md North Star). It:

  1. retrieves the top-k chunks from Qdrant (M5.2 `search`),
  2. renders them as numbered passages, and
  3. asks the shared LLM writer (`llm/client.py`) for an answer grounded ONLY on
     that context.

It returns the answer *and* the retrieved context, so the demo (M10) and eval
(M9) can show what the model was actually given.

As of M8.2 the LLM call goes through the one reusable writer in `llm/client.py`:
the same model, decoding, and strict "answer only from context" template used by
the hybrid pipeline (`retrieval/pipeline.py`) and the graph compare glue
(`graph/compare.py`). The ONLY thing that differs between this baseline and
Graph-RAG is the retrieved context — which is exactly the contrast the demo
makes.

    python src/vector/rag.py "What is the Transformer architecture?"
    python src/vector/rag.py --k 8 "Which papers study retrieval-augmented generation?"

Uses: src/vector/search.py (M5.2), src/llm/client.py (M8.1).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import client as llm_client  # noqa: E402
from vector import search  # noqa: E402


def format_context(contexts: list[dict]) -> str:
    """Render retrieved chunks as numbered [n] Title / text blocks for the prompt."""
    blocks = []
    for i, c in enumerate(contexts, 1):
        title = c.get("title", "")
        text = c.get("text", "")
        blocks.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(blocks)


def vector_rag(query: str, k: int = 5, client=None, model: str | None = None,
               host: str | None = None) -> dict:
    """Baseline vector RAG: retrieve -> ground -> answer.

    Returns {query, answer, contexts} where contexts are the retrieved chunks
    (payload + score) that grounded the answer. `client` is an optional Qdrant
    client (the LLM writer is the shared `llm/client.py`).
    """
    contexts = search.search(query, k=k, client=client)
    context = format_context(contexts)
    answer = llm_client.answer_from_context(query, context, model=model, host=host)
    return {"query": query, "answer": answer, "contexts": contexts}


def main() -> int:
    args = sys.argv[1:]
    k = 5
    if "--k" in args:
        i = args.index("--k")
        k = int(args[i + 1])
        del args[i : i + 2]
    if not args:
        print('usage: python src/vector/rag.py [--k N] "your question"', file=sys.stderr)
        return 1
    query = " ".join(args)

    result = vector_rag(query, k=k)
    print(f"Q: {result['query']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for rank, c in enumerate(result["contexts"], 1):
        print(f"  [{rank}] ({c['score']:.4f}) {c.get('title', '')}  "
              f"({c.get('paper_id')}#{c.get('chunk_index')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
