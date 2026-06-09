"""Baseline vector-RAG pipeline: retrieve -> ground -> answer (M5.3).

`vector_rag(query)` is the traditional-RAG baseline the Graph-RAG engine has to
beat on multi-hop questions (CLAUDE.md North Star). It:

  1. retrieves the top-k chunks from Qdrant (M5.2 `search`),
  2. builds a context-grounded prompt (answer ONLY from the passages), and
  3. asks the local Ollama model to write the answer.

It returns the answer *and* the retrieved context, so the demo (M10) and eval
(M9) can show what the model was actually given.

The LLM call here is deliberately minimal — the reusable Ollama wrapper and the
hardened "answer only from context" template are M8, which will replace
`generate()` and rewire both pipelines through one writer (M8.2). Keeping it
local for now lets the baseline run end-to-end without pre-building M8.

    python src/vector/rag.py "What is the Transformer architecture?"
    python src/vector/rag.py --k 8 "Which papers study retrieval-augmented generation?"

Uses: src/vector/search.py (M5.2), Ollama (config.OLLAMA_MODEL).
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from vector import search  # noqa: E402

SYSTEM_PROMPT = (
    "You are a research-paper assistant. Answer the question using ONLY the "
    "numbered context passages below. If the context does not contain the answer, "
    "say you don't know — do not use outside knowledge. Be concise and refer to "
    "paper titles where helpful."
)

# qwen3 and other reasoning models may emit a <think>...</think> trace; strip it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def format_context(contexts: list[dict]) -> str:
    """Render retrieved chunks as numbered [n] Title / text blocks for the prompt."""
    blocks = []
    for i, c in enumerate(contexts, 1):
        title = c.get("title", "")
        text = c.get("text", "")
        blocks.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(query: str, contexts: list[dict]) -> str:
    """Assemble the grounded user prompt (context passages + question)."""
    ctx = format_context(contexts) if contexts else "(no context retrieved)"
    return f"Context passages:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"


def generate(prompt: str, model: str | None = None, host: str | None = None,
             system: str = SYSTEM_PROMPT) -> str:
    """Minimal Ollama chat call (temperature 0); reasoning traces stripped."""
    import ollama

    client = ollama.Client(host=host or config.OLLAMA_HOST)
    resp = client.chat(
        model=model or config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )
    return _strip_think(resp.message.content)


def vector_rag(query: str, k: int = 5, client=None, model: str | None = None,
               host: str | None = None) -> dict:
    """Baseline vector RAG: retrieve -> ground -> answer.

    Returns {query, answer, contexts} where contexts are the retrieved chunks
    (payload + score) that grounded the answer.
    """
    contexts = search.search(query, k=k, client=client)
    prompt = build_prompt(query, contexts)
    answer = generate(prompt, model=model, host=host)
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
