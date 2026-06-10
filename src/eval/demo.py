"""M9.4: The single most dramatic multi-hop demo question.

Runs both pipelines side-by-side on the selected demo question and prints a
formatted comparison showing exactly where Vector RAG fails and Graph-RAG nails it.

The winner is multihop-001 (collaborating_institutions template, gap=1.000):
  - Graph: context_entity_recall=1.000 (all 5 gold institutions retrieved)
  - Vector: context_entity_recall=0.000 (zero gold institutions in chunks)

Usage:
    python src/eval/demo.py              # run live (requires Neo4j + Qdrant + Ollama)
    python src/eval/demo.py --cached     # print saved results from demo_question.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

DEMO_PATH = os.path.join(config.DATA_PROCESSED, "demo_question.json")

# The demo question (frozen by M9.4 analysis — do not change; changing it would
# invalidate the benchmark comparison used in M11's LinkedIn post).
DEMO_QUESTION = "Which institutions collaborate most with Tsinghua University?"


def _wrap(text: str, width: int = 78, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def run_live() -> dict:
    """Run both pipelines and return a result dict for caching."""
    from eval.harness import deterministic_scores  # noqa: PLC0415
    from graph.compare import graph_rag  # noqa: PLC0415
    from vector.rag import vector_rag  # noqa: PLC0415

    print("Running vector RAG...", file=sys.stderr)
    vr = vector_rag(DEMO_QUESTION, k=5)
    print("Running graph RAG...", file=sys.stderr)
    gr = graph_rag(DEMO_QUESTION, limit=15)

    gold = [
        "University of Hong Kong",
        "Chinese University of Hong Kong",
        "Shanghai Artificial Intelligence Laboratory",
        "Hong Kong University of Science and Technology",
        "Beijing Academy of Artificial Intelligence",
    ]

    v_ctx = [f"{c.get('title', '')}\n{c.get('text', '')}".strip()
             for c in vr["contexts"]]
    g_ctx = ([", ".join(f"{kk}={vv}" for kk, vv in row.items()) for row in gr["rows"]]
             if gr["rows"] else [gr["context"]])

    v_rec = {"answer_entities": gold, "contexts": v_ctx, "answer": vr["answer"]}
    g_rec = {"answer_entities": gold, "contexts": g_ctx, "answer": gr["answer"]}
    v_rec.update(deterministic_scores(v_rec))
    g_rec.update(deterministic_scores(g_rec))

    return {
        "question": DEMO_QUESTION,
        "gold_entities": gold,
        "vector": {
            "answer": vr["answer"],
            "contexts": v_ctx,
            "context_entity_recall": v_rec["context_entity_recall"],
            "answer_entity_recall": v_rec["answer_entity_recall"],
        },
        "graph": {
            "answer": gr["answer"],
            "contexts": g_ctx,
            "template": gr["template"],
            "seed_name": gr.get("seed_name"),
            "context_entity_recall": g_rec["context_entity_recall"],
            "answer_entity_recall": g_rec["answer_entity_recall"],
        },
        "gap": g_rec["context_entity_recall"] - v_rec["context_entity_recall"],
    }


def _print_comparison(result: dict) -> None:
    q = result["question"]
    v = result["vector"]
    g = result["graph"]
    gold = result["gold_entities"]

    bar = "=" * 80
    half = "-" * 80

    print(f"\n{bar}")
    print("  DEMO QUESTION (M9.4 — most dramatic multi-hop win)")
    print(bar)
    print(_wrap(q))
    print(f"\n  Gold entities: {', '.join(gold)}")
    print(f"\n  Context recall:  Vector={v['context_entity_recall']:.3f}  "
          f"Graph={g['context_entity_recall']:.3f}  "
          f"(gap={result['gap']:+.3f})")
    print(f"  Answer recall:   Vector={v['answer_entity_recall']:.3f}  "
          f"Graph={g['answer_entity_recall']:.3f}")

    print(f"\n{half}")
    print("  VECTOR RAG — FAILS")
    print(half)
    print("  Retrieved (top-3 chunks):")
    for i, c in enumerate(v["contexts"][:3], 1):
        preview = c.replace("\n", " ")[:100]
        print(f"  [{i}] {preview}")
    print(f"\n  Answer:\n{_wrap(v['answer'])}")

    print(f"\n{half}")
    print(f"  GRAPH-RAG — NAILS IT  (template: {g['template']})")
    print(half)
    print("  Retrieved (graph traversal rows):")
    for i, c in enumerate(g["contexts"][:5], 1):
        print(f"  [{i}] {c}")
    print(f"\n  Answer:\n{_wrap(g['answer'])}")
    print(f"\n{bar}\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Run or display the M9.4 demo question.")
    p.add_argument("--cached", action="store_true",
                   help="print results from demo_question.json (no live services needed)")
    p.add_argument("--save", action="store_true", default=True,
                   help="save live results to demo_question.json (default: on)")
    args = p.parse_args()

    if args.cached:
        if not os.path.exists(DEMO_PATH):
            print(f"No cached result at {DEMO_PATH}. Run without --cached first.",
                  file=sys.stderr)
            return 1
        with open(DEMO_PATH, encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = run_live()
        if args.save:
            os.makedirs(os.path.dirname(DEMO_PATH), exist_ok=True)
            with open(DEMO_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved -> {DEMO_PATH}", file=sys.stderr)

    _print_comparison(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
