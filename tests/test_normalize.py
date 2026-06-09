"""Unit tests for entity-name normalization (M3.2)."""

import json

from ingest import normalize


def test_strip_country_only_for_real_countries():
    assert normalize.strip_country("Google (United States)") == ("Google", True)
    assert normalize.strip_country("China University of Geosciences (Beijing)") == (
        "China University of Geosciences (Beijing)",
        False,
    )
    assert normalize.strip_country("Northeastern University") == ("Northeastern University", False)


def test_canonical_name_map():
    assert normalize.canonical_name("Google (United Kingdom)") == "Google"
    assert normalize.canonical_name("open ai") == "OpenAI"
    assert normalize.canonical_name("OpenAI") == "OpenAI"


def test_slug():
    assert normalize.slug("Robert Bosch") == "ORG:robert-bosch"
    assert normalize.slug("Ernst & Young") == "ORG:ernst-young"


def _paper(pid, insts):
    return {
        "paper_id": pid, "title": "t", "abstract": "", "has_abstract": False,
        "text": "t", "year": 2024, "cited_by_count": 1,
        "authors": [{"author_id": "A" + pid, "name": "X", "institutions": insts}],
        "topics": [], "references": [],
    }


def test_merges_country_splits_but_not_ambiguous(tmp_path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        _paper("1", [{"inst_id": "I_us", "name": "Google (United States)", "ror": None}]),
        _paper("2", [{"inst_id": "I_uk", "name": "Google (United Kingdom)", "ror": None}]),
        # same identical raw name, two ids -> must stay separate
        _paper("3", [{"inst_id": "I_a", "name": "Northeastern University", "ror": None}]),
        _paper("4", [{"inst_id": "I_b", "name": "Northeastern University", "ror": None}]),
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    stats = normalize.normalize(str(inp), str(out))
    res = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    g1 = res[0]["authors"][0]["institutions"][0]
    g2 = res[1]["authors"][0]["institutions"][0]
    assert g1["inst_id"] == g2["inst_id"] == "ORG:google"  # merged
    assert g1["name"] == "Google"
    n3 = res[2]["authors"][0]["institutions"][0]["inst_id"]
    n4 = res[3]["authors"][0]["institutions"][0]["inst_id"]
    assert n3 != n4  # ambiguous same-name kept separate
    assert stats["orgs_merged"] == 1
