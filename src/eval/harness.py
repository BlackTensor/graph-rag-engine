"""Evaluation harness: Vector-only RAG vs Graph-RAG, scored with Ragas (M9.2).

Runs both retrieval pipelines over the frozen `eval/questions.jsonl` (M9.1) and
scores them so M9.3 can table the contrast. Two metric families:

* **Ragas** (LLM-judged, fully local — no paid API): faithfulness,
  answer_correctness (accuracy), context_recall, llm_context_precision. The
  judge is the same local Ollama model the engine answers with; the embeddings
  are the project's bge-small (reused, so nothing extra to pull).
* **Deterministic entity-recall** (judge-free): the fraction of a question's
  gold `answer_entities` that appear (a) in the retrieved context and (b) in the
  answer. This is the robust backbone of the comparison — it needs no LLM, is
  fully reproducible, and is exactly the kind of LLM-independent signal M6.3
  used. It cleanly captures the North Star claim: on multi-hop questions the
  graph surfaces the gold entities (high context recall) while the vector index
  — title+abstract chunks only — structurally can't.

Cost note: generation is the bottleneck (one local LLM call per question per
pipeline, slow on CPU). Generations are cached to `data/processed/eval_runs.jsonl`
and reused across runs; use `--limit N` for a quick subset and `--no-ragas` to
skip the (slow) LLM judge and get the deterministic numbers immediately.

    python src/eval/harness.py --limit 5                 # 5 easy + 5 multi-hop, full scoring
    python src/eval/harness.py --no-ragas                # all 100, deterministic only (fast)
    python src/eval/harness.py --pipelines vector,graph  # the two arms (default)
    python src/eval/harness.py --regen                   # ignore the generation cache

Uses: vector.rag (M5.3), graph.compare (M6.3), eval/questions.jsonl (M9.1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

QUESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eval", "questions.jsonl",
)
RUNS_CACHE = os.path.join(config.DATA_PROCESSED, "eval_runs.jsonl")
SCORES_CSV = os.path.join(config.DATA_PROCESSED, "eval_scores.csv")

# The Ragas metrics we report, mapped to M9.3's "accuracy / precision / recall /
# faithfulness" table. Names are the Ragas metric `.name` values.
RAGAS_METRICS = [
    "faithfulness",
    "answer_correctness",
    "context_recall",
    "llm_context_precision_with_reference",
]

# Per-judge-call timeout (s). Generous because the judge is a small local model
# on CPU (~30-40s/call), and metrics like faithfulness chain several calls.
RAGAS_TIMEOUT = 900


# --- questions ----------------------------------------------------------------

def load_questions(path: str = QUESTIONS_PATH, limit: int | None = None,
                   split: str = "all") -> list[dict]:
    """Load the eval set; `limit` is per-split, `split` filters easy/multi_hop."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if limit is not None:
        out, counts = [], {}
        for r in rows:
            c = counts.get(r["split"], 0)
            if c < limit:
                out.append(r)
                counts[r["split"]] = c + 1
        rows = out
    return rows


# --- pipelines ----------------------------------------------------------------
# Each adapter runs a retrieval+answer pipeline and returns the common shape
# {answer, contexts} where contexts is a list[str] (Ragas `retrieved_contexts`).

def _vector_pipeline(query: str, k: int = 5) -> dict:
    from vector import rag  # noqa: PLC0415

    res = rag.vector_rag(query, k=k)
    contexts = [
        f"{c.get('title', '')}\n{c.get('text', '')}".strip() for c in res["contexts"]
    ]
    return {"answer": res["answer"], "contexts": contexts, "diag": {}}


def _graph_pipeline(query: str, k: int = 15) -> dict:
    from graph import compare  # noqa: PLC0415

    res = compare.graph_rag(query, limit=k)
    if res["rows"]:
        contexts = [", ".join(f"{kk}={vv}" for kk, vv in row.items()) for row in res["rows"]]
    else:
        contexts = [res["context"]]  # abstention message
    return {
        "answer": res["answer"],
        "contexts": contexts,
        "diag": {"template": res["template"], "seed_name": res.get("seed_name")},
    }


PIPELINES = {
    "vector": _vector_pipeline,
    "graph": _graph_pipeline,
}


# --- generation (cached) ------------------------------------------------------

def _cache_key(qid: str, pipeline: str) -> str:
    return f"{qid}::{pipeline}"


def _load_cache(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                cache[_cache_key(r["id"], r["pipeline"])] = r
    return cache


def generate(questions: list[dict], pipelines: list[str], k: int = 5,
             cache_path: str = RUNS_CACHE, regen: bool = False) -> list[dict]:
    """Run each pipeline over each question; cache to JSONL and reuse.

    Returns one run record per (question, pipeline) with the answer, retrieved
    contexts, and the question's gold fields carried through for scoring.
    """
    cache = {} if regen else _load_cache(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    records: list[dict] = []
    # Append-only cache write so a crash mid-run keeps what's done.
    mode = "w" if regen else "a"
    with open(cache_path, mode, encoding="utf-8") as cf:
        for q in questions:
            for name in pipelines:
                key = _cache_key(q["id"], name)
                if key in cache:
                    records.append(cache[key])
                    continue
                out = PIPELINES[name](q["question"], k=k)
                rec = {
                    "id": q["id"],
                    "split": q["split"],
                    "pipeline": name,
                    "question": q["question"],
                    "ground_truth": q["ground_truth"],
                    "answer_entities": q["answer_entities"],
                    "expected_route": q["expected_route"],
                    "graph_template": q["graph_template"],
                    "answer": out["answer"],
                    "contexts": out["contexts"],
                    "diag": out["diag"],
                }
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cf.flush()
                cache[key] = rec
                records.append(rec)
                print(f"  generated {key}", file=sys.stderr)
    return records


# --- deterministic entity-recall (judge-free) ---------------------------------

def _entity_hit(entity: str, text: str) -> bool:
    return entity.lower() in text.lower()


def deterministic_scores(rec: dict) -> dict:
    """Judge-free metrics from the gold `answer_entities` (fraction found)."""
    ents = rec["answer_entities"]
    if not ents:
        return {"context_entity_recall": float("nan"), "answer_entity_recall": float("nan")}
    ctx = "\n".join(rec["contexts"])
    ans = rec["answer"]
    ctx_hits = sum(_entity_hit(e, ctx) for e in ents)
    ans_hits = sum(_entity_hit(e, ans) for e in ents)
    return {
        "context_entity_recall": ctx_hits / len(ents),
        "answer_entity_recall": ans_hits / len(ents),
    }


# --- Ragas scoring (local LLM judge + bge embeddings) -------------------------

def _install_ragas_shim() -> None:
    """ragas 0.4.3 hard-imports a chat model path langchain-community 1.x dropped.

    Inject a stub so `import ragas` succeeds; we never use ChatVertexAI (the
    judge is local Ollama). Must run before importing ragas.
    """
    import types  # noqa: PLC0415

    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = stub


def _bge_embeddings():
    """LangChain Embeddings backed by the project's bge-small (reused, L2-norm)."""
    from langchain_core.embeddings import Embeddings  # noqa: PLC0415

    from vector.search import _get_model  # noqa: PLC0415

    class _BGE(Embeddings):
        def __init__(self):
            self.m = _get_model(config.EMBEDDING_MODEL)

        def embed_documents(self, texts):
            return self.m.encode(list(texts), normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self.m.encode([text], normalize_embeddings=True)[0].tolist()

    return _BGE()


def _build_judge(metric_names: list[str]):
    """Build (metrics, llm, embeddings) for Ragas, all local."""
    _install_ragas_shim()
    from langchain_ollama import ChatOllama  # noqa: PLC0415
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
    from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
    from ragas.metrics import (  # noqa: PLC0415
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    # reasoning=False disables qwen3's <think> traces, which otherwise break the
    # judge's structured-output parsing.
    chat = ChatOllama(
        model=config.OLLAMA_MODEL, base_url=config.OLLAMA_HOST,
        temperature=0, reasoning=False,
    )
    llm = LangchainLLMWrapper(chat)
    emb = LangchainEmbeddingsWrapper(_bge_embeddings())

    by_name = {
        "faithfulness": Faithfulness(),
        "answer_correctness": AnswerCorrectness(),
        "context_recall": LLMContextRecall(),
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
    }
    metrics = [by_name[n] for n in metric_names if n in by_name]
    return metrics, llm, emb


def ragas_scores(records: list[dict], metric_names: list[str]) -> dict[str, dict]:
    """Score the run records with Ragas; return {record_index: {metric: value}}.

    Keyed by position in `records` (one EvaluationDataset row each). Per-sample
    failures yield NaN (raise_exceptions=False) so the run never hard-crashes on
    a flaky local-judge parse.
    """
    _install_ragas_shim()
    from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: PLC0415
    from ragas.run_config import RunConfig  # noqa: PLC0415

    metrics, llm, emb = _build_judge(metric_names)
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)
    # Serialize (max_workers=1) — Ragas would otherwise fire ~16 concurrent judge
    # calls at the single CPU-bound Ollama, where they queue and each blows past
    # the default 180s job timeout. With a long per-call timeout and no
    # concurrency, the LLM-heavy metrics (faithfulness, answer_correctness) get
    # real values instead of TimeoutError -> NaN.
    run_config = RunConfig(timeout=RAGAS_TIMEOUT, max_workers=1)
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=emb,
                      run_config=run_config, raise_exceptions=False, show_progress=True)
    df = result.to_pandas()
    out: dict[int, dict] = {}
    for i in range(len(records)):
        row = df.iloc[i]
        out[i] = {m.name: float(row.get(m.name, float("nan"))) for m in metrics}
    return out


# --- aggregation + output -----------------------------------------------------

def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(scored: list[dict], metric_cols: list[str]) -> list[dict]:
    """Mean of each metric per (pipeline, split) and per (pipeline, overall)."""
    rows: list[dict] = []
    pipelines = sorted({r["pipeline"] for r in scored})
    splits = sorted({r["split"] for r in scored}) + ["all"]
    for pipe in pipelines:
        for split in splits:
            subset = [
                r for r in scored
                if r["pipeline"] == pipe and (split == "all" or r["split"] == split)
            ]
            if not subset:
                continue
            row = {"pipeline": pipe, "split": split, "n": len(subset)}
            for col in metric_cols:
                row[col] = _mean([r.get(col, float("nan")) for r in subset])
            rows.append(row)
    return rows


def _write_csv(scored: list[dict], metric_cols: list[str], path: str) -> None:
    import csv  # noqa: PLC0415

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["id", "split", "pipeline", *metric_cols]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in scored:
            w.writerow({c: r.get(c) for c in cols})


def _print_table(agg: list[dict], metric_cols: list[str]) -> None:
    headers = ["pipeline", "split", "n", *metric_cols]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}" if v == v else "nan"
        return str(v)

    widths = {h: max(len(h), *(len(fmt(r.get(h, ""))) for r in agg)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for r in agg:
        print("  ".join(fmt(r.get(h, "")).ljust(widths[h]) for h in headers))


# --- CLI ----------------------------------------------------------------------

def run(limit: int | None, pipelines: list[str], metric_names: list[str],
        k: int, regen: bool, use_ragas: bool, split: str) -> list[dict]:
    """Full harness: load -> generate -> score -> aggregate. Returns scored recs."""
    questions = load_questions(limit=limit, split=split)
    print(f"Evaluating {len(questions)} questions x {len(pipelines)} pipelines "
          f"({', '.join(pipelines)})", file=sys.stderr)
    records = generate(questions, pipelines, k=k, regen=regen)

    det_cols = ["context_entity_recall", "answer_entity_recall"]
    for r in records:
        r.update(deterministic_scores(r))

    metric_cols = list(det_cols)
    if use_ragas:
        rag_out = ragas_scores(records, metric_names)
        for i, r in enumerate(records):
            r.update(rag_out.get(i, {}))
        metric_cols += [m for m in metric_names if m in RAGAS_METRICS]

    _write_csv(records, metric_cols, SCORES_CSV)
    agg = aggregate(records, metric_cols)
    print()
    _print_table(agg, metric_cols)
    print(f"\nPer-question scores -> {SCORES_CSV}", file=sys.stderr)
    return records


def main() -> int:
    p = argparse.ArgumentParser(description="Vector vs Graph-RAG eval (Ragas + deterministic).")
    p.add_argument("--limit", type=int, default=None, help="questions per split")
    p.add_argument("--pipelines", default="vector,graph")
    p.add_argument("--metrics", default=",".join(RAGAS_METRICS))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--split", default="all", choices=["all", "easy", "multi_hop"])
    p.add_argument("--regen", action="store_true", help="ignore generation cache")
    p.add_argument("--no-ragas", action="store_true", help="deterministic metrics only")
    args = p.parse_args()

    run(
        limit=args.limit,
        pipelines=[s.strip() for s in args.pipelines.split(",") if s.strip()],
        metric_names=[s.strip() for s in args.metrics.split(",") if s.strip()],
        k=args.k,
        regen=args.regen,
        use_ragas=not args.no_ragas,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
