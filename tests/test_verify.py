"""Unit tests for graph verification (M4.3).

No live Neo4j: covers the pure `expected_counts` recomputation, which is what
gives the verifier independent ground truth to compare the graph against.
"""

from graph import verify


def _rec(pid, authors, topics, refs):
    return {
        "paper_id": pid,
        "authors": authors,
        "topics": topics,
        "references": refs,
    }


def test_expected_counts_uses_distinct_merge_semantics():
    records = [
        _rec(
            "W1",
            authors=[
                {"author_id": "A1", "institutions": [{"inst_id": "I1"}]},
                {"author_id": "A2", "institutions": [{"inst_id": "I1"}]},
            ],
            topics=[{"topic_id": "T1"}, {"topic_id": "T2"}],
            refs=["W2"],  # in-corpus
        ),
        _rec(
            "W2",
            authors=[{"author_id": "A1", "institutions": [{"inst_id": "I2"}]}],
            topics=[{"topic_id": "T1"}],
            refs=["W404"],  # out-of-corpus -> dropped
        ),
    ]
    c = verify.expected_counts(records)
    assert c["Paper"] == 2
    assert c["Author"] == 2  # A1, A2 distinct
    assert c["Institution"] == 2  # I1, I2
    assert c["Topic"] == 2  # T1, T2
    assert c["AUTHORED_BY"] == 3  # (W1,A1),(W1,A2),(W2,A1)
    assert c["WORKS_AT"] == 3  # (A1,I1),(A2,I1),(A1,I2)
    assert c["STUDIES"] == 3  # (W1,T1),(W1,T2),(W2,T1)
    assert c["CITES"] == 1  # only W1->W2; W2->W404 dropped


def test_expected_counts_dedupes_repeated_pairs():
    # Same author listed twice on one paper collapses to one AUTHORED_BY edge.
    records = [
        _rec(
            "W1",
            authors=[
                {"author_id": "A1", "institutions": [{"inst_id": "I1"}]},
                {"author_id": "A1", "institutions": [{"inst_id": "I1"}]},
            ],
            topics=[],
            refs=[],
        )
    ]
    c = verify.expected_counts(records)
    assert c["AUTHORED_BY"] == 1
    assert c["WORKS_AT"] == 1


def test_sample_queries_present_for_browser():
    assert "schema" in verify.SAMPLE_QUERIES
    assert all(q.strip() for q in verify.SAMPLE_QUERIES.values())
