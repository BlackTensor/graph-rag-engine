"""Unit tests for cleaning transforms (no data file needed)."""

from ingest import clean


def test_reconstruct_abstract_orders_tokens():
    inv = {"Hello": [0], "graph": [2], "world": [1]}
    assert clean.reconstruct_abstract(inv) == "Hello world graph"
    assert clean.reconstruct_abstract(None) == ""
    assert clean.reconstruct_abstract({}) == ""


def test_parse_work_normalizes_and_filters_cites():
    raw = {
        "id": "https://openalex.org/W1",
        "title": "  A Paper  ",
        "abstract_inverted_index": {"hi": [0]},
        "publication_year": 2024,
        "cited_by_count": 5,
        "authorships": [
            {
                "author": {"id": "https://openalex.org/A9", "display_name": "Jane Doe"},
                "institutions": [
                    {"id": "https://openalex.org/I7", "display_name": "MIT", "ror": "r"}
                ],
            }
        ],
        "topics": [{"id": "https://openalex.org/T3", "display_name": "NLP",
                    "field": {"display_name": "Computer Science"}}],
        "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W999"],
    }
    rec = clean.parse_work(raw, corpus_ids={"W1", "W2"})
    assert rec["paper_id"] == "W1"
    assert rec["title"] == "A Paper"
    assert rec["abstract"] == "hi" and rec["has_abstract"] is True
    assert rec["authors"][0]["author_id"] == "A9"
    assert rec["authors"][0]["institutions"][0]["inst_id"] == "I7"
    assert rec["topics"][0]["topic_id"] == "T3"
    assert rec["references"] == ["W2"]  # W999 dropped (out of corpus)


def test_parse_work_abstract_falls_back_to_title():
    raw = {"id": "https://openalex.org/W5", "title": "Only Title",
           "abstract_inverted_index": None}
    rec = clean.parse_work(raw, corpus_ids={"W5"})
    assert rec["has_abstract"] is False
    assert rec["text"] == "Only Title"
