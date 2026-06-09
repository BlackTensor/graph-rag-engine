"""Wiring tests for the M9.2 eval harness (no live services).

Exercises the pure machinery — question loading/limiting, the generation loop +
JSONL cache (resume), the judge-free deterministic entity-recall, and
aggregation — by monkeypatching the pipeline adapters so no Ollama/Qdrant/Neo4j
is touched. The Ragas judge itself is integration-tested live, not here.
"""

from __future__ import annotations

import math

import pytest

from eval import harness


@pytest.fixture
def fake_pipelines(monkeypatch):
    """Replace the real pipelines with deterministic stubs."""
    def vec(query, k=5):
        return {"answer": f"vector answer to {query}",
                "contexts": ["Tsinghua University and friends"], "diag": {}}

    def graph(query, k=15):
        return {
            "answer": "University of Hong Kong leads",
            "contexts": ["institution=University of Hong Kong, shared_papers=14"],
            "diag": {"template": "collaborating_institutions",
                     "seed_name": "Tsinghua University"},
        }

    monkeypatch.setitem(harness.PIPELINES, "vector", vec)
    monkeypatch.setitem(harness.PIPELINES, "graph", graph)


def test_load_questions_limit_per_split():
    rows = harness.load_questions(limit=3)
    assert len([r for r in rows if r["split"] == "easy"]) == 3
    assert len([r for r in rows if r["split"] == "multi_hop"]) == 3


def test_load_questions_split_filter():
    rows = harness.load_questions(split="multi_hop")
    assert rows and all(r["split"] == "multi_hop" for r in rows)


def test_deterministic_scores_fraction():
    rec = {
        "answer_entities": ["University of Hong Kong", "CUHK", "Missing Org"],
        "contexts": ["institution=University of Hong Kong", "institution=CUHK"],
        "answer": "The top collaborator is the University of Hong Kong.",
    }
    s = harness.deterministic_scores(rec)
    assert s["context_entity_recall"] == pytest.approx(2 / 3)  # 2 of 3 entities in context
    assert s["answer_entity_recall"] == pytest.approx(1 / 3)   # only UHK named in answer


def test_deterministic_scores_empty_entities_is_nan():
    s = harness.deterministic_scores({"answer_entities": [], "contexts": [], "answer": ""})
    assert math.isnan(s["context_entity_recall"])


def test_generate_writes_and_resumes_cache(tmp_path, fake_pipelines):
    cache = tmp_path / "runs.jsonl"
    qs = harness.load_questions(limit=1)  # 1 easy + 1 multi_hop
    recs = harness.generate(qs, ["vector", "graph"], cache_path=str(cache), regen=True)
    assert len(recs) == len(qs) * 2
    n_lines = sum(1 for _ in open(cache, encoding="utf-8"))
    assert n_lines == len(recs)

    # Second call (regen=False) must reuse the cache, not append duplicates.
    calls = {"n": 0}
    orig = harness.PIPELINES["vector"]

    def counting(q, k=5):
        calls["n"] += 1
        return orig(q, k)

    harness.PIPELINES["vector"] = counting
    try:
        recs2 = harness.generate(qs, ["vector", "graph"], cache_path=str(cache), regen=False)
    finally:
        harness.PIPELINES["vector"] = orig
    assert calls["n"] == 0, "cached runs should not re-invoke the pipeline"
    assert len(recs2) == len(recs)
    assert sum(1 for _ in open(cache, encoding="utf-8")) == n_lines


def test_aggregate_means_per_pipeline_split():
    scored = [
        {"pipeline": "graph", "split": "multi_hop", "context_entity_recall": 1.0},
        {"pipeline": "graph", "split": "multi_hop", "context_entity_recall": 0.5},
        {"pipeline": "vector", "split": "multi_hop", "context_entity_recall": 0.0},
    ]
    agg = harness.aggregate(scored, ["context_entity_recall"])
    graph_multi = next(r for r in agg if r["pipeline"] == "graph" and r["split"] == "multi_hop")
    assert graph_multi["context_entity_recall"] == pytest.approx(0.75)
    assert graph_multi["n"] == 2
    # an "all" rollup row exists per pipeline
    assert any(r["split"] == "all" for r in agg)


def test_ragas_shim_allows_import():
    """The shim must make `import ragas` succeed under langchain-community 1.x."""
    harness._install_ragas_shim()
    pytest.importorskip("ragas")  # skip only if ragas truly isn't installed
