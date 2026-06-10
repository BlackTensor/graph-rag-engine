"""Tests for the M9.3 results table + chart module (no live services)."""

from __future__ import annotations

import json

import pytest

from eval import results

# --- fixtures -----------------------------------------------------------------

def _make_record(pid, split, pipeline, ctx_ents, ctx_hits, ans_hits):
    """Build a fake eval record pre-populated with entity-recall scores."""
    ents = [f"Org{i}" for i in range(ctx_ents)]
    ctx = " ".join(ents[:ctx_hits])
    ans = " ".join(ents[:ans_hits])
    return {
        "id": pid,
        "split": split,
        "pipeline": pipeline,
        "question": "q",
        "ground_truth": "gt",
        "answer_entities": ents,
        "contexts": [ctx],
        "answer": ans,
        "diag": {},
    }


SAMPLE_RECORDS = [
    # easy: vector good, graph bad
    _make_record("e1", "easy", "vector", 2, 2, 2),
    _make_record("e2", "easy", "vector", 2, 2, 1),
    _make_record("e1", "easy", "graph", 2, 0, 0),
    _make_record("e2", "easy", "graph", 2, 1, 0),
    # multi_hop: graph good, vector bad
    _make_record("m1", "multi_hop", "vector", 3, 0, 0),
    _make_record("m2", "multi_hop", "vector", 3, 1, 0),
    _make_record("m1", "multi_hop", "graph", 3, 3, 2),
    _make_record("m2", "multi_hop", "graph", 3, 3, 3),
]


# --- load_records -------------------------------------------------------------

def test_load_records_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RUNS_CACHE", str(tmp_path / "nope.jsonl"))
    monkeypatch.setattr(results, "RETRIEVAL_CACHE", str(tmp_path / "nope2.jsonl"))
    recs, label = results.load_records("auto")
    assert recs == []
    assert label == "none"


def test_load_records_reads_retrieval_first(tmp_path, monkeypatch):
    rpath = tmp_path / "retrieval_eval.jsonl"
    rpath.write_text(json.dumps({"id": "x", "pipeline": "graph"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(results, "RETRIEVAL_CACHE", str(rpath))
    monkeypatch.setattr(results, "RUNS_CACHE", str(tmp_path / "nope.jsonl"))
    recs, label = results.load_records("auto")
    assert len(recs) == 1
    assert label == "retrieval-only"


# --- score_records ------------------------------------------------------------

def test_score_records_adds_deterministic_keys():
    recs = [_make_record("e1", "easy", "vector", 2, 2, 1)]
    results.score_records(recs)
    assert recs[0]["context_entity_recall"] == pytest.approx(1.0)
    assert recs[0]["answer_entity_recall"] == pytest.approx(0.5)


# --- present_metrics ----------------------------------------------------------

def test_present_metrics_excludes_all_nan():
    recs = [{"context_entity_recall": 0.5, "faithfulness": float("nan")}]
    cols = results._present_metrics(recs)
    assert "context_entity_recall" in cols
    assert "faithfulness" not in cols


# --- build_markdown_table -----------------------------------------------------

def test_build_markdown_table_structure():
    from eval.harness import aggregate
    results.score_records(SAMPLE_RECORDS)
    metric_cols = results._present_metrics(SAMPLE_RECORDS)
    agg = aggregate(SAMPLE_RECORDS, metric_cols)
    md = results.build_markdown_table(agg, metric_cols)
    # Must have a header row and separator row and at least one data row
    lines = [ln for ln in md.splitlines() if ln.strip()]
    assert lines[0].startswith("|")
    assert "---" in lines[1]
    assert len(lines) >= 3


def test_build_markdown_table_north_star_values():
    """Graph should beat vector on multi-hop context recall."""
    from eval.harness import aggregate
    results.score_records(SAMPLE_RECORDS)
    metric_cols = ["context_entity_recall"]
    agg = aggregate(SAMPLE_RECORDS, metric_cols)
    graph_mh = next(r for r in agg if r["pipeline"] == "graph" and r["split"] == "multi_hop")
    vector_mh = next(r for r in agg if r["pipeline"] == "vector" and r["split"] == "multi_hop")
    assert graph_mh["context_entity_recall"] > vector_mh["context_entity_recall"]


# --- write_table + draw_chart --------------------------------------------------

def test_write_table_creates_file(tmp_path, monkeypatch):
    from eval.harness import aggregate
    results.score_records(SAMPLE_RECORDS)
    metric_cols = results._present_metrics(SAMPLE_RECORDS)
    agg = aggregate(SAMPLE_RECORDS, metric_cols)
    out = tmp_path / "results_table.md"
    results.write_table(agg, metric_cols, "test-source", path=str(out))
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Graph-RAG" in content
    assert "context_entity_recall" in content.lower() or "Context Recall" in content


def test_draw_chart_creates_png(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from eval.harness import aggregate
    results.score_records(SAMPLE_RECORDS)
    metric_cols = results._present_metrics(SAMPLE_RECORDS)
    agg = aggregate(SAMPLE_RECORDS, metric_cols)
    out = tmp_path / "chart.png"
    results.draw_chart(agg, path=str(out))
    assert out.exists()
    assert out.stat().st_size > 1000  # non-empty PNG


# --- retrieval-only harness integration ---------------------------------------

def test_retrieval_only_cache_separate(monkeypatch):
    """RETRIEVAL_CACHE and RUNS_CACHE are distinct paths."""
    from eval import harness
    assert harness.RETRIEVAL_CACHE != harness.RUNS_CACHE


def test_retrieval_pipelines_registered():
    from eval import harness
    assert "vector" in harness.RETRIEVAL_PIPELINES
    assert "graph" in harness.RETRIEVAL_PIPELINES
    assert harness.RETRIEVAL_PIPELINES is not harness.PIPELINES
