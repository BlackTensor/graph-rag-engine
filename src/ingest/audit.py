"""Schema sanity-check / coverage audit for the raw OpenAlex pull (M2.3).

Confirms every field our graph needs is present (or flags it for derivation in
M3), and measures in-corpus citation density (drives multi-hop quality in M4).

    python src/ingest/audit.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def short_id(openalex_url: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123'."""
    return openalex_url.rsplit("/", 1)[-1] if openalex_url else None


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(n: int, total: int) -> str:
    return f"{n:>6} / {total} ({100 * n / total:5.1f}%)" if total else "0"


def audit(path: str) -> None:
    works = load(path)
    total = len(works)
    ids = {short_id(w.get("id")) for w in works}
    print(f"Records: {total}  |  distinct paper ids: {len(ids)}  "
          f"(duplicates: {total - len(ids)})\n")

    # --- per-paper field coverage ---
    has_title = sum(bool(w.get("title")) for w in works)
    has_abstract = sum(bool(w.get("abstract_inverted_index")) for w in works)
    has_year = sum(w.get("publication_year") is not None for w in works)
    print("PAPER fields")
    print(f"  title              {pct(has_title, total)}")
    print(f"  abstract (inv idx) {pct(has_abstract, total)}   <- if missing, use title (M3)")
    print(f"  publication_year   {pct(has_year, total)}\n")

    # --- authors + institutions ---
    authors, insts = set(), set()
    papers_with_author = authorships_total = authorships_with_inst = 0
    for w in works:
        a = w.get("authorships") or []
        if a:
            papers_with_author += 1
        for au in a:
            authorships_total += 1
            aid = (au.get("author") or {}).get("id")
            if aid:
                authors.add(aid)
            ai = au.get("institutions") or []
            if ai:
                authorships_with_inst += 1
            for inst in ai:
                if inst.get("id"):
                    insts.add(inst["id"])
    print("AUTHOR / INSTITUTION")
    print(f"  papers w/ >=1 author     {pct(papers_with_author, total)}")
    print(f"  distinct authors         {len(authors)}")
    print(f"  authorships w/ >=1 inst   {pct(authorships_with_inst, authorships_total)}"
          f"   <- WORKS_AT coverage; rest have no affiliation")
    print(f"  distinct institutions    {len(insts)}\n")

    # --- topics ---
    topics, fields = set(), {}
    papers_with_topic = 0
    for w in works:
        t = w.get("topics") or []
        if t:
            papers_with_topic += 1
        for tp in t:
            if tp.get("id"):
                topics.add(tp["id"])
            f = (tp.get("field") or {}).get("display_name")
            if f:
                fields[f] = fields.get(f, 0) + 1
    print("TOPIC")
    print(f"  papers w/ >=1 topic      {pct(papers_with_topic, total)}")
    print(f"  distinct topics          {len(topics)}")
    top_fields = sorted(fields.items(), key=lambda kv: -kv[1])[:6]
    print(f"  top fields               {top_fields}\n")

    # --- citations (CITES) + in-corpus density ---
    refs_total = in_corpus = 0
    papers_with_ref = papers_with_in_ref = 0
    cited_in_corpus = set()
    for w in works:
        r = w.get("referenced_works") or []
        if r:
            papers_with_ref += 1
        has_in = False
        for ref in r:
            refs_total += 1
            rid = short_id(ref)
            if rid in ids:
                in_corpus += 1
                cited_in_corpus.add(rid)
                has_in = True
        if has_in:
            papers_with_in_ref += 1
    print("CITES (citation edges)")
    print(f"  papers w/ >=1 reference  {pct(papers_with_ref, total)}")
    print(f"  total reference edges    {refs_total}")
    print(f"  in-corpus edges          {pct(in_corpus, refs_total)}   <- usable CITES")
    print(f"  papers citing in-corpus  {pct(papers_with_in_ref, total)}")
    print(f"  papers cited in-corpus   {pct(len(cited_in_corpus), total)}")


def main() -> int:
    path = os.path.join(config.DATA_RAW, "openalex", "works.jsonl")
    if not os.path.exists(path):
        print(f"missing {path} -- run src/ingest/download.py first", file=sys.stderr)
        return 1
    audit(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
