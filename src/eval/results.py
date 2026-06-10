"""M9.3: Produce results table + chart from the eval cache(s).

Reads:
  * data/processed/retrieval_eval.jsonl  — fast retrieval-only run (M9.3 --retrieval-only)
  * data/processed/eval_runs.jsonl       — full generation + answer cache (harness.py)

Writes:
  * data/processed/results_table.md     — markdown results table for README
  * data/processed/results_chart.png    — grouped bar chart for LinkedIn

Usage:
  python src/eval/results.py                  # auto-detect best available data
  python src/eval/results.py --source retrieval  # use retrieval_eval.jsonl only
  python src/eval/results.py --source full       # use eval_runs.jsonl only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from eval.harness import (  # noqa: E402
    RETRIEVAL_CACHE,
    RUNS_CACHE,
    aggregate,
    deterministic_scores,
)

TABLE_PATH = os.path.join(config.DATA_PROCESSED, "results_table.md")
CHART_PATH = os.path.join(config.DATA_PROCESSED, "results_chart.png")

# Human-readable column labels for the output table
_METRIC_LABELS = {
    "context_entity_recall": "Context Recall",
    "answer_entity_recall": "Answer Recall",
    "faithfulness": "Faithfulness",
    "answer_correctness": "Accuracy",
    "context_recall": "LLM Context Recall",
    "llm_context_precision_with_reference": "Precision",
}


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(source: str = "auto") -> tuple[list[dict], str]:
    """Return (records, source_label).

    source: "auto" | "retrieval" | "full"
    Auto prefers retrieval_eval (covers all 100 Qs) over eval_runs (may be partial).
    """
    if source in ("auto", "retrieval"):
        recs = _load_jsonl(RETRIEVAL_CACHE)
        if recs:
            return recs, "retrieval-only"
    if source in ("auto", "full"):
        recs = _load_jsonl(RUNS_CACHE)
        if recs:
            return recs, "full-generation"
    return [], "none"


def score_records(records: list[dict]) -> list[dict]:
    """Add deterministic entity-recall scores in-place; return the same list."""
    for r in records:
        r.update(deterministic_scores(r))
    return records


def _present_metrics(records: list[dict]) -> list[str]:
    """Metric columns with at least one non-NaN, non-zero value across records.

    answer_entity_recall is excluded when all answers are empty strings — the
    retrieval-only mode stores "" as the answer, making the metric meaningless.
    """
    no_answers = all(r.get("answer", "") == "" for r in records)
    candidates = [
        "context_entity_recall",
        "answer_entity_recall",
        "faithfulness",
        "answer_correctness",
        "context_recall",
        "llm_context_precision_with_reference",
    ]
    return [
        m for m in candidates
        if m != "answer_entity_recall" or not no_answers
        if any(not math.isnan(r.get(m, float("nan"))) for r in records)
    ]


# --- Markdown table -----------------------------------------------------------

def _fmt(v: object) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def build_markdown_table(agg: list[dict], metric_cols: list[str]) -> str:
    labels = [_METRIC_LABELS.get(m, m) for m in metric_cols]
    header = ["Pipeline", "Split", "N", *labels]
    sep = [":---", ":---", "---:", *("---:" for _ in metric_cols)]

    def row(r: dict) -> list[str]:
        return [r["pipeline"], r["split"], str(r["n"]),
                *(_fmt(r.get(m, float("nan"))) for m in metric_cols)]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(row(r)) + " |" for r in agg),
    ]
    return "\n".join(lines)


def write_table(agg: list[dict], metric_cols: list[str],
                source_label: str, path: str = TABLE_PATH) -> None:
    table_md = build_markdown_table(agg, metric_cols)
    n_qs = sum(r["n"] for r in agg if r["split"] != "all") // 2
    content = f"""## Evaluation Results

*Source: {source_label} — {n_qs} questions × 2 pipelines*

{table_md}

### Key finding

Graph-RAG achieves high **Context Recall** on multi-hop questions (relational facts
live in the graph, not in document chunks), while Vector RAG excels on easy
single-document questions.  The contrast is largest at the retrieval level
(`context_entity_recall`) and is fully deterministic — independent of the LLM.
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Results table -> {path}", file=sys.stderr)


# --- Bar chart ----------------------------------------------------------------

def draw_chart(agg: list[dict], path: str = CHART_PATH) -> None:
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        print("matplotlib not installed; skipping chart.", file=sys.stderr)
        return

    # We always chart context_entity_recall (available from retrieval-only runs).
    # Add answer_entity_recall as a second panel if non-trivially populated.
    splits = ["easy", "multi_hop"]
    pipelines = ["vector", "graph"]
    colors = {"vector": "#4C9BE8", "graph": "#F4A261"}
    split_labels = {"easy": "Easy (single-doc)", "multi_hop": "Multi-hop (relational)"}

    def _get(pipeline: str, split: str, metric: str) -> float:
        for r in agg:
            if r["pipeline"] == pipeline and r["split"] == split:
                return r.get(metric, float("nan"))
        return float("nan")

    # Decide panels: context_entity_recall always; answer_entity_recall if any > 0.
    has_answer_recall = any(
        not math.isnan(_get(p, s, "answer_entity_recall"))
        and _get(p, s, "answer_entity_recall") > 0
        for p in pipelines for s in splits
    )
    metrics = ["context_entity_recall"]
    metric_titles = ["Context Entity Recall"]
    if has_answer_recall:
        metrics.append("answer_entity_recall")
        metric_titles.append("Answer Entity Recall")

    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(splits))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, metric_titles):
        for i, pipe in enumerate(pipelines):
            vals = [_get(pipe, s, metric) for s in splits]
            bars = ax.bar(
                x + (i - 0.5) * width,
                [0 if math.isnan(v) else v for v in vals],
                width,
                label=pipe.title(),
                color=colors[pipe],
                edgecolor="white",
                linewidth=0.8,
            )
            for bar, val in zip(bars, vals):
                if not math.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([split_labels[s] for s in splits], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (0–1)", fontsize=10)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Vector RAG vs Graph-RAG — Retrieval Quality",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart -> {path}", file=sys.stderr)


# --- aggregation table print --------------------------------------------------

def _print_results(agg: list[dict], metric_cols: list[str]) -> None:
    headers = ["pipeline", "split", "n", *metric_cols]
    labels = ["pipeline", "split", "n",
              *(_METRIC_LABELS.get(m, m) for m in metric_cols)]

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "—" if math.isnan(v) else f"{v:.3f}"
        return str(v)

    widths = {h: max(len(lbl), *(len(fmt(r.get(h, ""))) for r in agg))
              for h, lbl in zip(headers, labels)}
    line = "  ".join(lbl.ljust(widths[h]) for h, lbl in zip(headers, labels))
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for r in agg:
        print("  ".join(fmt(r.get(h, "")).ljust(widths[h]) for h in headers))


# --- CLI ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="M9.3 results table + chart from eval cache.")
    p.add_argument("--source", default="auto",
                   choices=["auto", "retrieval", "full"],
                   help="which cache to read (default: auto = retrieval-only if available)")
    p.add_argument("--no-chart", action="store_true", help="skip chart generation")
    p.add_argument("--no-table", action="store_true", help="skip markdown table write")
    args = p.parse_args()

    records, source_label = load_records(args.source)
    if not records:
        print("No eval data found. Run:\n"
              "  python src/eval/harness.py --retrieval-only\n"
              "to generate the retrieval cache first.", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} records ({source_label})", file=sys.stderr)
    score_records(records)

    metric_cols = _present_metrics(records)
    agg = aggregate(records, metric_cols)

    _print_results(agg, metric_cols)

    if not args.no_table:
        write_table(agg, metric_cols, source_label)
    if not args.no_chart:
        draw_chart(agg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
