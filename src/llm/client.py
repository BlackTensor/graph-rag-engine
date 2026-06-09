"""Reusable Ollama wrapper + strict "answer ONLY from context" template (M8.1).

This is the single answer-writer the whole engine routes through. Today the
vector baseline (`vector/rag.py`, M5.3), the graph compare glue (`graph/compare.py`,
M6.3), and the hybrid pipeline (`retrieval/pipeline.py`, M7.3) each carry their
own copy of the Ollama call + a grounding prompt. M8.1 centralizes that glue
here; M8.2 rewires those callers through this module so there is exactly one
writer and one hardened template.

What's centralized:
  * the Ollama chat call — config-driven model/host, deterministic
    (temperature 0 + fixed seed, CLAUDE.md §7), reasoning traces (`<think>…`)
    stripped;
  * one strict "answer ONLY from the context" system prompt that handles the
    merged context (knowledge-graph facts and/or document passages);
  * the grounded prompt builder;
  * a clear `LLMError` so the demo (M10) / eval (M9) get a useful message when
    Ollama is down instead of a raw driver traceback.

    python src/llm/client.py "Your question?" --context "facts the model may use"

Uses: Ollama (config.OLLAMA_MODEL / config.OLLAMA_HOST).
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# Deterministic decoding so the same context yields the same answer (eval/demo
# reproducibility, CLAUDE.md §7). qwen3 et al. honour both options.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 0

# Canonical refusal. Emitting one exact string lets the eval harness (M9) detect
# abstention deterministically instead of pattern-matching free-form "I don't know"s.
INSUFFICIENT_CONTEXT = "I don't know based on the available context."

# The strict template. It must (a) forbid outside knowledge, (b) explain that the
# context can mix knowledge-graph facts with document passages, (c) tell the model
# to prefer the precise graph facts for relational/ranking questions, and (d) give
# it one canonical way to say it can't answer.
SYSTEM_PROMPT = (
    "You are a research assistant for an AI-research-papers knowledge base. "
    "Answer the user's question using ONLY the information in the context below.\n\n"
    "Rules:\n"
    "- Use ONLY facts that appear in the context. Never use prior or outside "
    "knowledge, and never invent titles, names, numbers, or relationships.\n"
    "- The context may contain knowledge-graph facts (exact relationships, "
    "affiliations, citations, and ranked counts) and/or document passages (a "
    "paper's title and abstract).\n"
    "- For questions about who works with or at whom, who cites whom, or which "
    "entities rank highest, rely on the knowledge-graph facts and report the "
    "ranked items with their counts.\n"
    f"- If the context does not contain the answer, reply exactly: "
    f'"{INSUFFICIENT_CONTEXT}" Do not guess.\n'
    "- Be concise and factual; refer to paper titles where helpful."
)

# qwen3 and other reasoning models may emit a <think>...</think> trace; strip it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when the Ollama call cannot complete (service down, model missing)."""


def strip_think(text: str) -> str:
    """Remove a model's <think>...</think> reasoning trace and surrounding space."""
    return _THINK_RE.sub("", text).strip()


def build_prompt(query: str, context: str) -> str:
    """Assemble the grounded user prompt from a (merged) context block + question."""
    ctx = context.strip() if context and context.strip() else "(no context retrieved)"
    return f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"


def generate(
    prompt: str,
    *,
    system: str = SYSTEM_PROMPT,
    model: str | None = None,
    host: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> str:
    """Low-level Ollama chat call: deterministic, reasoning trace stripped.

    Raises `LLMError` (not a raw driver exception) if Ollama is unreachable or
    the model/response fails, so callers can surface a clean message.
    """
    import ollama

    client = ollama.Client(host=host or config.OLLAMA_HOST)
    try:
        resp = client.chat(
            model=model or config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature, "seed": seed},
        )
    except ollama.ResponseError as e:  # model missing, bad request, etc.
        raise LLMError(f"Ollama responded with an error: {e}") from e
    except Exception as e:  # connection refused, timeout (httpx errors, etc.)
        raise LLMError(
            f"Could not reach Ollama at {host or config.OLLAMA_HOST}: {e}"
        ) from e
    return strip_think(resp.message.content)


def answer_from_context(
    query: str,
    context: str,
    *,
    model: str | None = None,
    host: str | None = None,
    system: str = SYSTEM_PROMPT,
) -> str:
    """Answer `query` grounded ONLY on `context` (the strict template).

    The one call every pipeline will use (M8.2): build the grounded prompt and
    run it through the hardened system prompt.
    """
    return generate(
        build_prompt(query, context), system=system, model=model, host=host
    )


def main() -> int:
    args = sys.argv[1:]
    context = ""
    if "--context" in args:
        i = args.index("--context")
        context = args[i + 1]
        del args[i : i + 2]
    if not args:
        print(
            'usage: python src/llm/client.py "question" [--context "facts..."]',
            file=sys.stderr,
        )
        return 1
    query = " ".join(args)

    try:
        print(answer_from_context(query, context))
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
