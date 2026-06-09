# M2.1 — Data Source Decision

**Status:** decided (2026-06-09) · **Owner task:** M2.1

## Decision

Use **[OpenAlex](https://openalex.org)** as the single data source, scoped to a
**focused NLP / LLM subfield** of recent, well-cited papers (~5–6k works).

- **Source:** OpenAlex Works API — `https://api.openalex.org/works`
- **License:** **CC0 1.0** (public domain) — no usage restrictions for the demo.
- **Access:** free, **no API key**. Use the *polite pool* by adding
  `mailto=shayanx13@gmail.com` to every request (higher, more stable rate limits:
  ~10 req/s, 100k/day).

### Corpus definition (the reproducible filter)

```
filter = concepts.id:C204321447,from_publication_date:2024-01-01,cited_by_count:>10
```

- `concepts.id:C204321447` → **Natural Language Processing** (the focused subfield)
- `from_publication_date:2024-01-01` → recent (current, LinkedIn-relevant)
- `cited_by_count:>10` → trims noise, keeps a **citation-dense** core for strong
  multi-hop chains

**Verified count (2026-06-09): 5,847 works** — within the 5–10k target.

> ⚠️ OpenAlex is a living dataset, so this count drifts over time. M2.2 will
> **cache the raw pull** to `data/raw/` and record the capture date so the build is
> reproducible from the snapshot, not re-fetched.

## Why OpenAlex (over the alternatives)

| Source | Has all 5 (Paper/Author/Inst/Topic/Cite) + text? | Verdict |
|---|---|---|
| **OpenAlex** | ✅ all present; authors + institutions disambiguated (canonical **ROR** IDs), canonical topics, `referenced_works` for citations | **Chosen** — cleanest entities = least normalization; API gives a tight, controllable subset (no giant download) |
| AMiner DBLP v14 | ✅ but `org` is free-text (heavy normalization); ~16 GB download to filter | Rejected — too heavy + messy for v1 |
| Semantic Scholar | ⚠️ affiliations spotty → weak `WORKS_AT` / institution multi-hop | Rejected — institution coverage is core to our demo |
| arXiv (Kaggle) | ❌ no institutions, no citation edges | Rejected — missing two core relationships |

## Field → graph schema mapping

Pulled via `select=` to keep payloads small. Each Work maps as:

| Graph element | OpenAlex field |
|---|---|
| `Paper` | `id` (strip `https://openalex.org/` prefix), `title`, `abstract` (reconstructed), `publication_year`, `cited_by_count` |
| `Author` + `AUTHORED_BY` | `authorships[].author` (`id`, `display_name`) → Paper |
| `Institution` + `WORKS_AT` | `authorships[].institutions[]` (`id`, `display_name`, `ror`) → Author |
| `Topic` + `STUDIES` | `topics[]` (`id`, `display_name`, `field`) ← Paper |
| `CITES` | `referenced_works[]` (Paper → Paper) |

### Two known transforms (handled in M2.2 / M3)

1. **Abstracts are inverted indexes.** `abstract_inverted_index` is
   `{token: [positions]}`; reconstruct to plain text by ordering tokens by position.
   (Needed for vector RAG.)
2. **Citation density.** `referenced_works` often points outside our subset. For a
   self-contained, dense `CITES` graph we'll **keep edges where both endpoints are in
   the corpus**; optionally backfill a few high-impact external targets as
   abstract-less `Paper` stubs. (Final call in M4.)

## Fetch plan (for M2.2)

- Endpoint: `GET /works` with the filter above + `mailto`.
- Paginate with **cursor** (`cursor=*`), `per-page=200` → ~30 requests for ~5.8k.
- `select=id,title,publication_year,cited_by_count,abstract_inverted_index,authorships,topics,referenced_works`.
- Cache raw JSON pages to `data/raw/openalex/` (don't re-download if present).
