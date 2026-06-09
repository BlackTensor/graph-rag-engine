"""Quick quality/speed bench to pick the production answer-model (M8.3).

Holds the *context* fixed and varies only the Ollama model, so this is a clean
model-only comparison (no retrieval noise) that needs only Ollama running — not
Neo4j/Qdrant. Each case runs through the real shared writer (`llm/client.py`,
M8.1/M8.2), so we measure exactly what the engine will do in production.

Three cases mirror the engine's actual workload:
  1. definitional   — answer from a document passage (the vector route),
  2. relational     — answer/rank from knowledge-graph facts (the graph route),
  3. abstention     — context lacks the answer; the model must refuse, not guess.

Per model it reports each answer's latency + a pass/fail quality check, then the
per-model pass count and mean latency. With a speed priority (CLAUDE.md M8.3),
the pick is the FASTEST model that passes all three.

    python src/llm/benchmark.py                                  # default candidates
    python src/llm/benchmark.py --models qwen3:0.6b qwen3:1.7b   # explicit set
    python src/llm/benchmark.py --repeat 3                       # average N runs/case

Uses: src/llm/client.py (M8.1/M8.2), Ollama.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from llm import client  # noqa: E402

DEFAULT_MODELS = ["qwen3:0.6b", "qwen3:1.7b"]


@dataclass
class Case:
    name: str
    query: str
    context: str
    check: Callable[[str], bool]  # passes if True (answer lowercased)
    note: str


def _refused(ans: str) -> bool:
    return "don't know" in ans or "do not know" in ans


CASES: list[Case] = [
    Case(
        name="definitional",
        query="Which architecture is based solely on attention, dispensing with "
        "recurrence and convolutions?",
        context=(
            "[1] Attention Is All You Need\nWe propose the Transformer, a model "
            "architecture based solely on attention mechanisms, dispensing with "
            "recurrence and convolutions entirely."
        ),
        check=lambda a: "transformer" in a and not _refused(a),
        note="must name the Transformer from the passage",
    ),
    Case(
        name="relational",
        query="Which institutions collaborate most with Tsinghua University?",
        # Mirrors the real merged graph context (graph/retrieve.render_context):
        # a labelled head + template description + numbered rows.
        context=(
            "Knowledge-graph facts:\n"
            "Graph traversal `collaborating_institutions` "
            '(seed institution = "Tsinghua University")\n'
            "Institutions that co-author the most papers with the seed institution.\n"
            "  1. institution=University of Hong Kong, shared_papers=14\n"
            "  2. institution=Chinese University of Hong Kong, shared_papers=13\n"
            "  3. institution=Shanghai AI Laboratory, shared_papers=12"
        ),
        check=lambda a: "hong kong" in a and not _refused(a),
        note="must report the top-ranked institution from the graph facts",
    ),
    Case(
        name="abstention",
        query="What is the H-index of the author Ada Lovelace?",
        context=(
            "[1] Attention Is All You Need\nWe propose the Transformer, a model "
            "based solely on attention mechanisms."
        ),
        check=_refused,
        note="context lacks the answer -> must refuse, not hallucinate",
    ),
]


def run_model(model: str, repeat: int = 1) -> dict:
    """Run every case `repeat` times through `model`; return per-case results."""
    results = []
    for case in CASES:
        latencies, last_answer, ok = [], "", False
        for _ in range(repeat):
            t0 = time.perf_counter()
            answer = client.answer_from_context(case.query, case.context, model=model)
            latencies.append(time.perf_counter() - t0)
            last_answer = answer
            ok = case.check(answer.lower())
        results.append(
            {
                "case": case.name,
                "ok": ok,
                "latency": sum(latencies) / len(latencies),
                "answer": last_answer,
            }
        )
    passed = sum(r["ok"] for r in results)
    mean_latency = sum(r["latency"] for r in results) / len(results)
    return {"model": model, "results": results, "passed": passed,
            "mean_latency": mean_latency}


def pick(summaries: list[dict]) -> tuple[dict, str]:
    """Choose the winner: fastest model passing every case, else most-correct.

    Speed priority (CLAUDE.md M8.3): among models that pass all cases, take the
    lowest mean latency; if none pass all, fall back to most cases passed then
    fastest. Returns (winner_summary, basis_string).
    """
    full = [s for s in summaries if s["passed"] == len(CASES)]
    pool = full or summaries
    pool = sorted(pool, key=lambda s: (-s["passed"], s["mean_latency"]))
    basis = "fastest with full quality" if full else "most correct (none passed all)"
    return pool[0], basis


def main() -> int:
    args = sys.argv[1:]
    models = DEFAULT_MODELS
    repeat = 1
    if "--models" in args:
        i = args.index("--models")
        models = []
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            models.append(args[j])
            j += 1
        del args[i:j]
    if "--repeat" in args:
        i = args.index("--repeat")
        repeat = int(args[i + 1])
        del args[i : i + 2]

    print(f"Ollama: {config.OLLAMA_HOST}  |  models: {', '.join(models)}  "
          f"|  repeat: {repeat}\n")
    summaries = []
    for model in models:
        print(f"=== {model} ===")
        try:
            summary = run_model(model, repeat=repeat)
        except client.LLMError as e:
            print(f"  SKIP ({e})\n")
            continue
        for r in summary["results"]:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['case']:<13} {r['latency']:6.2f}s  "
                  f"{r['answer'][:80].replace(chr(10), ' ')}")
        print(f"  -> {summary['passed']}/{len(CASES)} passed, "
              f"mean {summary['mean_latency']:.2f}s/case\n")
        summaries.append(summary)

    if not summaries:
        print("No models ran (Ollama down or none pulled).", file=sys.stderr)
        return 1

    # Speed priority: fastest model that passes every case; else most-correct.
    winner, basis = pick(summaries)
    print(f"PICK: {winner['model']}  "
          f"({winner['passed']}/{len(CASES)} passed, "
          f"{winner['mean_latency']:.2f}s/case)  [{basis}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
