"""Unit tests for the CSV export + data-quality report (M3.3)."""

import json

import pandas as pd

from ingest import export


def _paper(pid, insts, topics, refs, has_abstract=True):
    return {
        "paper_id": pid,
        "title": f"Title {pid}",
        "abstract": "abs" if has_abstract else f"Title {pid}",
        "has_abstract": has_abstract,
        "text": "abs",
        "year": 2024,
        "cited_by_count": 11,
        "authors": [
            {"author_id": "A" + pid, "name": "Jane, Doe", "institutions": insts}
        ],
        "topics": topics,
        "references": refs,
    }


def test_flatten_collapses_nested_and_dedupes_institutions():
    rec = _paper(
        "W1",
        insts=[
            {"inst_id": "I1", "name": "MIT", "ror": None},
            {"inst_id": "I1", "name": "MIT", "ror": None},  # dup -> collapsed
            {"inst_id": "I2", "name": "Stanford", "ror": None},
        ],
        topics=[{"topic_id": "T1", "name": "NLP", "field": "Computer Science"}],
        refs=["W2", "W3"],
    )
    row = export.flatten(rec)
    assert row["paper_id"] == "W1"
    assert row["n_authors"] == 1
    assert row["n_institutions"] == 2  # dup dropped
    assert row["institution_ids"] == "I1 | I2"
    assert row["institution_names"] == "MIT | Stanford"
    assert row["n_topics"] == 1
    assert row["topic_fields"] == "Computer Science"
    assert row["n_references"] == 2
    assert row["reference_ids"] == "W2 | W3"


def test_csv_roundtrip_preserves_commas_in_names(tmp_path):
    inp = tmp_path / "papers_normalized.jsonl"
    csv = tmp_path / "clean_papers.csv"
    report = tmp_path / "report.md"
    rec = _paper(
        "W1",
        insts=[{"inst_id": "I1", "name": "MIT", "ror": None}],
        topics=[{"topic_id": "T1", "name": "NLP", "field": "CS"}],
        refs=[],
    )
    inp.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    stats = export.export(str(inp), str(csv), str(report))
    assert stats["papers"] == 1

    back = pd.read_csv(csv)
    # "Jane, Doe" has a comma -> must survive CSV quoting.
    assert back.loc[0, "author_names"] == "Jane, Doe"
    assert report.read_text(encoding="utf-8").startswith("# Data-Quality Report")


def test_report_counts_distinct_entities(tmp_path):
    inp = tmp_path / "papers_normalized.jsonl"
    csv = tmp_path / "clean_papers.csv"
    report = tmp_path / "report.md"
    rows = [
        _paper("W1", [{"inst_id": "I1", "name": "MIT", "ror": None}],
               [{"topic_id": "T1", "name": "NLP", "field": "CS"}], ["W2"]),
        _paper("W2", [{"inst_id": "I1", "name": "MIT", "ror": None}],
               [{"topic_id": "T2", "name": "CV", "field": "CS"}], []),
    ]
    inp.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    export.export(str(inp), str(csv), str(report))
    text = report.read_text(encoding="utf-8")
    assert "**Institutions (distinct):** 1" in text  # shared I1
    assert "**Topics (distinct):** 2" in text
    assert "**CITES edges (in-corpus):** 1" in text
