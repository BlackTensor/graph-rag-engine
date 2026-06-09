"""Tests for the context-merge step (M7.2).

Pure logic — no DB, no LLM. Covers within-vector dedupe (one passage per paper),
cross-modal dedupe (drop a passage already named in the graph facts), the
labelled-section layout, and the empty case.
"""

from retrieval import merge


def _chunk(paper_id, score, title="T", chunk_index=0, text=None, **extra):
    return {
        "paper_id": paper_id,
        "chunk_id": f"{paper_id}-{chunk_index}",
        "chunk_index": chunk_index,
        "title": title,
        "score": score,
        "text": text if text is not None else f"{title}\n\nbody {chunk_index}",
        **extra,
    }


# --- dedupe -------------------------------------------------------------------

def test_dedupe_collapses_chunks_of_same_paper_keeping_best():
    # two chunks of W1 (score-descending) + one of W2 -> 2 passages, 1 dropped
    contexts = [
        _chunk("W1", 0.9, chunk_index=0),
        _chunk("W1", 0.7, chunk_index=1),
        _chunk("W2", 0.6),
    ]
    passages, dropped = merge.dedupe_passages(contexts)
    assert [p["paper_id"] for p in passages] == ["W1", "W2"]
    assert passages[0]["score"] == 0.9  # the best chunk of W1 is kept
    assert dropped == 1


def test_dedupe_drops_passage_named_in_graph():
    contexts = [_chunk("W1", 0.9, title="Attention Is All You Need")]
    passages, dropped = merge.dedupe_passages(
        contexts, graph_values=frozenset({"attention is all you need"})
    )
    assert passages == []
    assert dropped == 1


def test_dedupe_keeps_distinct_papers():
    contexts = [_chunk("W1", 0.9), _chunk("W2", 0.8), _chunk("W3", 0.7)]
    passages, dropped = merge.dedupe_passages(contexts)
    assert len(passages) == 3 and dropped == 0


# --- passage rendering --------------------------------------------------------

def test_format_passages_strips_duplicate_leading_title():
    # chunk 0 text repeats the title; the rendered body should not.
    c = _chunk("W1", 0.9, title="My Paper", text="My Paper\n\nthe abstract", year=2024,
               cited_by_count=12)
    out = merge.format_passages([c])
    assert "[1] My Paper (2024, 12 cites)" in out
    assert out.count("My Paper") == 1  # title not duplicated in the body
    assert "the abstract" in out


# --- merge_context ------------------------------------------------------------

def _graph_result(rows):
    return {
        "template": "collaborating_institutions",
        "seed_type": "institution",
        "seed_name": "Tsinghua University",
        "rows": rows,
        "context": "Graph traversal `collaborating_institutions` ...",
    }


def test_merge_both_modalities_labelled_sections():
    g = _graph_result([{"institution": "CUHK", "shared_papers": 13}])
    v = [_chunk("W1", 0.8, title="Some Survey")]
    merged = merge.merge_context(g, v)
    assert "Knowledge-graph facts:" in merged.text
    assert "Document passages:" in merged.text
    # graph section comes first (it's the precise relational answer)
    assert merged.text.index("Knowledge-graph") < merged.text.index("Document passages")
    assert merged.n_graph_rows == 1 and merged.n_passages == 1


def test_merge_graph_only_when_no_vector():
    merged = merge.merge_context(_graph_result([{"institution": "CUHK"}]), [])
    assert "Knowledge-graph facts:" in merged.text
    assert "Document passages:" not in merged.text


def test_merge_vector_only_when_graph_abstained():
    abstain = {"template": None, "rows": [], "context": "No graph traversal."}
    merged = merge.merge_context(abstain, [_chunk("W1", 0.8)])
    assert "Document passages:" in merged.text
    assert "Knowledge-graph facts:" not in merged.text
    assert merged.graph_facts is None


def test_merge_cross_modal_dedupe_counts_drop():
    g = _graph_result([{"paper": "Shared Paper", "cites": 99}])
    v = [_chunk("W1", 0.9, title="Shared Paper"), _chunk("W2", 0.8, title="Other")]
    merged = merge.merge_context(g, v)
    assert merged.n_passages == 1  # "Shared Paper" dropped (already in graph)
    assert merged.dropped == 1
    assert merged.passages[0]["title"] == "Other"


def test_merge_empty_when_nothing_retrieved():
    merged = merge.merge_context(None, [])
    assert merged.text == "(no context retrieved)"
    assert merged.n_passages == 0 and merged.n_graph_rows == 0
