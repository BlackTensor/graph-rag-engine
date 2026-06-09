"""Normalize entity names with a canonical-name map (M3.2).

Institutions are the main target. OpenAlex appends a country to many institution
display names (e.g. "Google (United States)") and splits global companies across
one id per country. We:

  1. Strip a trailing "(Country)" suffix (recognized countries only, so campus
     tags like "(Beijing)" are preserved).
  2. Apply an explicit canonical-name map for spelling variants
     (e.g. "open ai" -> "OpenAI") -- the extension point.
  3. Merge ids into one canonical org **only** when the collision is purely a
     country-suffix split (all variants were suffix-stripped). Identical raw
     names sharing multiple ids (e.g. "Northeastern University") are kept
     separate, since they are usually different institutions.

Reads data/interim/papers_clean.jsonl (M3.1), writes papers_normalized.jsonl.

    python src/ingest/normalize.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# Recognized country parentheticals (covers OpenAlex's country suffixes).
COUNTRIES = frozenset(
    {
        "United States", "United Kingdom", "China", "Canada", "Germany", "France",
        "Japan", "India", "Israel", "Switzerland", "Sweden", "Netherlands", "Italy",
        "Spain", "Australia", "Singapore", "South Korea", "Korea", "Finland", "Norway",
        "Denmark", "Belgium", "Austria", "Ireland", "Portugal", "Poland", "Romania",
        "Russia", "Brazil", "Mexico", "Taiwan", "Hong Kong", "Cayman Islands",
        "Saudi Arabia", "United Arab Emirates", "Turkey", "Greece", "Czech Republic",
        "Hungary", "New Zealand", "South Africa", "Iran", "Egypt", "Thailand",
        "Vietnam", "Indonesia", "Malaysia", "Pakistan", "Chile", "Argentina",
        "Colombia", "Luxembourg", "Estonia", "Slovenia", "Croatia", "Ukraine",
    }
)

# Explicit canonical-name overrides (lowercased stripped name -> canonical).
# Extend with known spelling variants. (The CLAUDE.md example.)
CANONICAL_NAME_MAP = {
    "open ai": "OpenAI",
    "openai": "OpenAI",
    "deepmind": "DeepMind",
    "google deepmind": "Google DeepMind",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
}


def _norm_ws(name: str) -> str:
    return " ".join((name or "").split())


def strip_country(name: str) -> tuple[str, bool]:
    """Return (name_without_trailing_country, had_country_suffix)."""
    m = re.search(r"\s*\(([^)]*)\)\s*$", name or "")
    if m and m.group(1).strip() in COUNTRIES:
        return name[: m.start()].strip(), True
    return _norm_ws(name), False


def canonical_name(raw: str) -> str:
    """Canonical institution display name (no merge decision)."""
    base, _ = strip_country(raw)
    return CANONICAL_NAME_MAP.get(base.lower(), base)


def slug(name: str) -> str:
    return "ORG:" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def normalize(in_path: str, out_path: str) -> dict:
    with open(in_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    # Pass 1: per canonical name, collect ids and whether any variant was bare
    # (i.e. had no country suffix -> ambiguous, do not merge).
    ids_by_name: dict[str, set] = {}
    bare_by_name: dict[str, bool] = {}
    for rec in records:
        for author in rec["authors"]:
            for inst in author["institutions"]:
                _, had_suffix = strip_country(inst["name"])
                cname = canonical_name(inst["name"])
                ids_by_name.setdefault(cname, set()).add(inst["inst_id"])
                bare_by_name[cname] = bare_by_name.get(cname, False) or (not had_suffix)

    mergeable = {
        name for name, ids in ids_by_name.items() if len(ids) > 1 and not bare_by_name[name]
    }

    # Pass 2: apply canonical names + merged ids.
    n_country_stripped = n_map_hits = 0
    ids_before, ids_after = set(), set()
    for rec in records:
        for author in rec["authors"]:
            new_insts, seen = [], set()
            for inst in author["institutions"]:
                base, had_suffix = strip_country(inst["name"])
                cname = canonical_name(inst["name"])
                ids_before.add(inst["inst_id"])
                if had_suffix:
                    n_country_stripped += 1
                if base.lower() in CANONICAL_NAME_MAP:
                    n_map_hits += 1
                cid = slug(cname) if cname in mergeable else inst["inst_id"]
                if cid in seen:  # merging can create dup inst within one author
                    continue
                seen.add(cid)
                ids_after.add(cid)
                new_insts.append({"inst_id": cid, "name": cname, "ror": inst.get("ror")})
            author["institutions"] = new_insts
            author["name"] = _norm_ws(author["name"])
        for topic in rec["topics"]:
            topic["name"] = _norm_ws(topic["name"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "papers": len(records),
        "institution_ids_before": len(ids_before),
        "institution_ids_after": len(ids_after),
        "orgs_merged": len(mergeable),
        "country_suffixes_stripped": n_country_stripped,
        "canonical_map_hits": n_map_hits,
    }


def main() -> int:
    in_path = os.path.join(config.DATA_INTERIM, "papers_clean.jsonl")
    out_path = os.path.join(config.DATA_INTERIM, "papers_normalized.jsonl")
    if not os.path.exists(in_path):
        print(f"missing {in_path} -- run src/ingest/clean.py first", file=sys.stderr)
        return 1
    stats = normalize(in_path, out_path)
    print(f"Wrote {stats['papers']} normalized papers -> {out_path}\n")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
