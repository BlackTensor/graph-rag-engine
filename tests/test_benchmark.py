"""Tests for the model bench harness (M8.3).

Quality checks, the run-loop wiring, and the speed-priority pick are covered
with pure logic / monkeypatching (no Ollama). The actual model comparison is a
live, one-off run done when picking the production model.
"""

from llm import benchmark


def test_cases_cover_the_three_routes():
    names = {c.name for c in benchmark.CASES}
    assert names == {"definitional", "relational", "abstention"}


def test_case_checks_pass_and_fail_as_expected():
    by_name = {c.name: c for c in benchmark.CASES}
    # definitional: must name the transformer, and not be a refusal
    assert by_name["definitional"].check("the transformer does this")
    assert not by_name["definitional"].check("i don't know based on the context")
    # relational: must surface the top institution
    assert by_name["relational"].check("mostly university of hong kong")
    assert not by_name["relational"].check("tsinghua university")
    # abstention: passes only when the model refuses
    assert by_name["abstention"].check("i don't know based on the available context.")
    assert not by_name["abstention"].check("her h-index is 42")


def test_run_model_records_latency_quality_and_answer(monkeypatch):
    # canned answers keyed by which query came in
    def fake_answer(query, context, model=None):
        if "attention" in query:
            return "The Transformer."
        if "collaborate" in query:
            return "University of Hong Kong leads."
        return "I don't know based on the available context."

    monkeypatch.setattr(benchmark.client, "answer_from_context", fake_answer)

    summary = benchmark.run_model("fake-model", repeat=1)
    assert summary["model"] == "fake-model"
    assert summary["passed"] == 3  # all three cases pass
    assert summary["mean_latency"] >= 0
    assert {r["case"] for r in summary["results"]} == {
        "definitional", "relational", "abstention"
    }


def test_pick_prefers_fastest_full_quality_model():
    summaries = [
        {"model": "slow-perfect", "passed": 3, "mean_latency": 20.0},
        {"model": "fast-perfect", "passed": 3, "mean_latency": 5.0},
        {"model": "fast-wrong", "passed": 1, "mean_latency": 1.0},
    ]
    winner, basis = benchmark.pick(summaries)
    assert winner["model"] == "fast-perfect"
    assert "fastest" in basis


def test_pick_falls_back_to_most_correct_when_none_pass_all():
    summaries = [
        {"model": "fast-weak", "passed": 1, "mean_latency": 2.0},
        {"model": "slow-better", "passed": 2, "mean_latency": 9.0},
    ]
    winner, basis = benchmark.pick(summaries)
    assert winner["model"] == "slow-better"  # more cases passed wins over speed
    assert "most correct" in basis
