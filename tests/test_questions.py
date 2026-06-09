"""Validate the M9.1 evaluation set `eval/questions.jsonl`.

A contract test for the frozen benchmark the M9.2 Ragas harness consumes: the
file exists, every line is valid JSON with the required schema, the split sizes
are exactly 50 + 50, ids/questions are unique, and the multi-hop entries name a
real M6.1 template. No live services required (pure file + schema checks).
"""

from __future__ import annotations

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(ROOT, "eval", "questions.jsonl")

REQUIRED_KEYS = {
    "id", "split", "question", "ground_truth", "answer_entities",
    "expected_route", "graph_template", "seed", "source",
}
VALID_ROUTES = {"vector", "graph", "hybrid"}


@pytest.fixture(scope="module")
def records() -> list[dict]:
    if not os.path.exists(QUESTIONS):
        pytest.skip("eval/questions.jsonl not built (run eval/build_questions.py)")
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_split_sizes(records):
    easy = [r for r in records if r["split"] == "easy"]
    multi = [r for r in records if r["split"] == "multi_hop"]
    assert len(easy) == 50
    assert len(multi) == 50
    assert len(records) == 100


def test_schema_and_uniqueness(records):
    ids, questions = set(), set()
    for r in records:
        assert REQUIRED_KEYS <= set(r), f"{r.get('id')}: missing keys"
        assert r["expected_route"] in VALID_ROUTES
        assert r["question"].strip(), f"{r['id']}: empty question"
        assert r["ground_truth"].strip(), f"{r['id']}: empty ground_truth"
        assert isinstance(r["answer_entities"], list) and r["answer_entities"], (
            f"{r['id']}: answer_entities must be a non-empty list"
        )
        assert r["id"] not in ids, f"duplicate id {r['id']}"
        assert r["question"] not in questions, f"duplicate question {r['id']}"
        ids.add(r["id"])
        questions.add(r["question"])


def test_split_specific_fields(records):
    for r in records:
        if r["split"] == "easy":
            assert r["graph_template"] is None
            assert r["seed"] is None
        else:
            assert isinstance(r["seed"], dict)
            assert r["seed"]["type"] in {"author", "institution", "topic"}
            assert r["seed"]["name"]


def test_multihop_templates_are_real(records):
    from graph import templates as T  # noqa: PLC0415

    for r in records:
        if r["split"] == "multi_hop":
            assert r["graph_template"] in T.TEMPLATES, (
                f"{r['id']}: unknown template {r['graph_template']!r}"
            )
