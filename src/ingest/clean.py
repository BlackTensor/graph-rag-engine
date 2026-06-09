"""Clean the raw OpenAlex pull: dedupe + missing-value handling (M3.1).

Parses raw Works into normalized paper records, dedupes by paper_id, reconstructs
abstracts (falling back to the title when absent), and keeps only **in-corpus**
CITES edges (decision (a): in-corpus only, no backfill -- CLAUDE.md §10).

Entity-name normalization is M3.2; the final clean_papers.csv + data-quality
report is M3.3. This step writes an interim JSONL (nested fields preserved).

    python src/ingest/clean.py

Writes: data/interim/papers_clean.jsonl
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from ingest.audit import short_id  # noqa: E402


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild plain text from OpenAlex's {token: [positions]} inverted index."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for token, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = token
    if not positions:
        return ""
    return " ".join(positions.get(i, "") for i in range(max(positions) + 1)).strip()


def parse_work(raw: dict, corpus_ids: set[str]) -> dict:
    pid = short_id(raw.get("id"))
    title = (raw.get("title") or "").strip()
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))

    authors, seen_a = [], set()
    for au in raw.get("authorships") or []:
        author = au.get("author") or {}
        aid = short_id(author.get("id"))
        if not aid or aid in seen_a:
            continue
        seen_a.add(aid)
        insts, seen_i = [], set()
        for inst in au.get("institutions") or []:
            iid = short_id(inst.get("id"))
            if not iid or iid in seen_i:
                continue
            seen_i.add(iid)
            insts.append(
                {"inst_id": iid, "name": (inst.get("display_name") or "").strip(),
                 "ror": inst.get("ror")}
            )
        authors.append(
            {"author_id": aid, "name": (author.get("display_name") or "").strip(),
             "institutions": insts}
        )

    topics, seen_t = [], set()
    for tp in raw.get("topics") or []:
        tid = short_id(tp.get("id"))
        if not tid or tid in seen_t:
            continue
        seen_t.add(tid)
        topics.append(
            {"topic_id": tid, "name": (tp.get("display_name") or "").strip(),
             "field": (tp.get("field") or {}).get("display_name")}
        )

    refs, seen_r = [], set()
    for ref in raw.get("referenced_works") or []:
        rid = short_id(ref)
        if rid in corpus_ids and rid != pid and rid not in seen_r:
            seen_r.add(rid)
            refs.append(rid)

    return {
        "paper_id": pid,
        "title": title,
        "abstract": abstract,
        "has_abstract": bool(abstract),
        "text": abstract or title,  # embedding text: abstract, else title
        "year": raw.get("publication_year"),
        "cited_by_count": raw.get("cited_by_count"),
        "authors": authors,
        "topics": topics,
        "references": refs,
    }


def clean(raw_path: str, out_path: str) -> dict:
    with open(raw_path, encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]

    corpus_ids = {short_id(w.get("id")) for w in raw}
    df = pd.DataFrame(parse_work(w, corpus_ids) for w in raw)

    n_input = len(df)
    df = df.drop_duplicates(subset="paper_id")
    n_dupes = n_input - len(df)

    before = len(df)
    df = df[df["paper_id"].notna() & (df["title"].str.len() > 0)]
    n_dropped_title = before - len(df)

    stats = {
        "input": n_input,
        "dropped_duplicate_paper_id": n_dupes,
        "dropped_missing_id_or_title": n_dropped_title,
        "output": len(df),
        "abstract_filled_from_title": int((~df["has_abstract"]).sum()),
        "papers_with_no_author": int((df["authors"].map(len) == 0).sum()),
        "in_corpus_cites_edges": int(df["references"].map(len).sum()),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    return stats


def main() -> int:
    raw_path = os.path.join(config.DATA_RAW, "openalex", "works.jsonl")
    out_path = os.path.join(config.DATA_INTERIM, "papers_clean.jsonl")
    if not os.path.exists(raw_path):
        print(f"missing {raw_path} -- run src/ingest/download.py first", file=sys.stderr)
        return 1
    stats = clean(raw_path, out_path)
    print(f"Wrote {stats['output']} cleaned papers -> {out_path}\n")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
