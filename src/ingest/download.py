"""Download the OpenAlex paper subset to data/raw/ (cached).

Corpus definition is documented in docs/data-source.md (M2.1).

    python src/ingest/download.py                  # full pull (skips if cached)
    python src/ingest/download.py --force          # re-download
    python src/ingest/download.py --max-records 50 # quick smoke pull

Writes (under data/raw/openalex/ by default):
    works.jsonl     one raw OpenAlex Work per line
    manifest.json   filter, capture timestamp, counts (for reproducibility)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# Allow running this file directly (python src/ingest/download.py): put the src/
# dir (parent of this package) on the path so `import config` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# --- Corpus definition (M2.1 / docs/data-source.md) ---
FILTER = "concepts.id:C204321447,from_publication_date:2024-01-01,cited_by_count:>10"
SELECT = ",".join(
    [
        "id",
        "title",
        "publication_year",
        "cited_by_count",
        "abstract_inverted_index",
        "authorships",
        "topics",
        "referenced_works",
    ]
)
PER_PAGE = 200
REQUEST_PAUSE_S = 0.2  # be polite to the API between pages


def build_url(cursor: str) -> str:
    query = urllib.parse.urlencode(
        {
            "filter": FILTER,
            "select": SELECT,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "mailto": config.OPENALEX_MAILTO,
        }
    )
    return f"{config.OPENALEX_BASE_URL}?{query}"


def _get_json(url: str, *, retries: int = 5) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": f"graphrag-discovery ({config.OPENALEX_MAILTO})"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < retries - 1:
                wait = 2**attempt
                print(f"  HTTP {exc.code}; retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def download(out_dir: str, *, force: bool = False, max_records: int | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    works_path = os.path.join(out_dir, "works.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")

    if os.path.exists(works_path) and os.path.exists(manifest_path) and not force:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        print(
            f"[cached] {manifest['n_records']} works in {works_path} "
            f"(captured {manifest['captured_at']}). Use --force to re-download."
        )
        return manifest

    tmp_path = works_path + ".tmp"
    cursor: str | None = "*"
    total: int | None = None
    n = pages = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        while cursor:
            data = _get_json(build_url(cursor))
            if total is None:
                total = data["meta"]["count"]
                print(f"Total matching works: {total}")
            results = data["results"]
            if not results:
                break
            for work in results:
                out.write(json.dumps(work, ensure_ascii=False) + "\n")
                n += 1
                if max_records and n >= max_records:
                    break
            pages += 1
            print(f"  page {pages}: +{len(results)} (total {n})")
            if max_records and n >= max_records:
                break
            cursor = data["meta"].get("next_cursor")
            time.sleep(REQUEST_PAUSE_S)

    os.replace(tmp_path, works_path)
    manifest = {
        "source": "openalex",
        "filter": FILTER,
        "select": SELECT,
        "api_total_count": total,
        "n_records": n,
        "n_pages": pages,
        "max_records": max_records,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] wrote {n} works to {works_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the OpenAlex paper subset.")
    parser.add_argument("--out", default=os.path.join(config.DATA_RAW, "openalex"))
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument(
        "--max-records", type=int, default=None, help="cap records (for a quick smoke pull)"
    )
    args = parser.parse_args(argv)
    download(args.out, force=args.force, max_records=args.max_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
