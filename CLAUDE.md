# CLAUDE.md — Graph-RAG Discovery Engine

> **Persistent project memory for Claude Code.**
> Read this file at the start of every session. Update the checkboxes as tasks complete. Never skip ahead — the user provides ONE task at a time. When a task is done, check its box, append a short note under it, and STOP.

---

## 0. How To Use This File (Rules for Claude)

1. **One task at a time.** The user will say "do M1.2" or "next task". Do only that. Do not start the next milestone unsolicited.
2. **Check the box when done.** Change `- [ ]` to `- [x]` for the completed sub-task. If all sub-tasks in a milestone are checked, also check the milestone header.
3. **Leave a breadcrumb.** Under each completed task, append a one-line `> ✅ Done: <what was built / file path / key decision>` note.
4. **Update the Status Log** (Section 9) at the end of every working session with date + what changed.
5. **Never delete checkboxes or rewrite history.** This file is the source of truth for progress.
6. **Confirm before destructive actions** (dropping the Neo4j DB, deleting datasets, overwriting checkpoints).
7. **Keep scope tight.** GNN is explicitly deferred to v2 (see Section 8). Do not build it in v1 unless the user re-prioritizes it.
8. **When uncertain, ask.** A 10-second question beats an hour of wrong work.

---

## 1. Project North Star

**Goal:** Build a Graph-RAG Discovery Engine over the **AI Research Papers** domain that beats traditional vector RAG on **multi-hop reasoning** questions, runs **fully local** (no API costs), and produces a **LinkedIn-ready demo**.

**The money shot:** A side-by-side where vector RAG *fails* a multi-hop question and Graph-RAG *nails* it.

**Domain:** AI Research Papers
**Entities:** `Paper`, `Author`, `Institution`, `Topic`
**Relationships:** `AUTHORED_BY`, `WORKS_AT`, `STUDIES`, `CITES`

**Scope discipline:** v1 ships WITHOUT a GNN. Neo4j native graph traversal handles multi-hop. GNN is a v2 stretch goal only.

---

## 2. Tech Stack (Locked)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Single language for v1 |
| Data wrangling | pandas, numpy | |
| Graph DB | Neo4j (Docker) | + `neo4j` python driver |
| Vector DB | Qdrant (Docker) | `qdrant-client` |
| Embeddings | `BAAI/bge-small-en-v1.5` | via sentence-transformers |
| Orchestration | LangGraph | query router + hybrid retrieval |
| Local LLM | Ollama | start with `qwen3` or `gemma` |
| Evaluation | Ragas | accuracy / faithfulness / recall |
| Demo UI | Streamlit | question box + answer + graph + docs |
| Dev tooling | Docker Compose, ruff, pytest | |

**Deferred to v2:** PyTorch Geometric, GCN/GAT, graph embeddings, MCP server.

---

## 3. Repository Layout (current — updated 2026-06-09)

> `[Mx]` = task that created it · `(pending Mx)` = not built yet. Working dir is
> `RAG Graph P`. Data files under `data/` are gitignored.

```
RAG Graph P/
├── CLAUDE.md                 # this file
├── README.md                 # [M1.4]
├── docker-compose.yml        # neo4j + qdrant + ollama  [M1.2]
├── pyproject.toml            # setuptools src-layout, ruff, pytest  [M1.1]
├── .env.example              # [M1.1; +OpenAlex M2.2; +DATA_INTERIM M3.1]
├── docs/
│   └── data-source.md        # OpenAlex decision + fetch plan  [M2.1]
├── data/
│   ├── raw/openalex/         # works.jsonl (55MB) + manifest.json  [M2.2]
│   ├── interim/              # papers_clean.jsonl [M3.1], papers_normalized.jsonl [M3.2]
│   └── processed/            # clean_papers.csv + data_quality_report.md [M3.3]; chunks.jsonl + embeddings.npy [M5.1]
├── src/
│   ├── config.py             # central settings (dotenv)  [M1.1]
│   ├── health.py             # connectivity smoke checks  [M1.3]
│   ├── ingest/               # download.py [M2.2], audit.py [M2.3], clean.py [M3.1], normalize.py [M3.2], export.py [M3.3]
│   ├── graph/                # neo4j build + queries  (pending M4)
│   ├── vector/               # embed.py [M5.1], index.py + search.py [M5.2], rag.py [M5.3]
│   ├── retrieval/            # hybrid + router (langgraph)  (pending M7)
│   ├── llm/                  # ollama wrapper  (pending M8)
│   └── eval/                 # ragas harness  (pending M9)
├── eval/
│   └── questions.jsonl       # 50 easy + 50 multi-hop  (pending M9.1)
├── app/
│   └── streamlit_app.py      # (pending M10)
└── tests/                    # scaffold, connections, download, audit, clean, normalize, export
```

---

## 4. Milestones & Tasks

> Format: `- [ ] ID — task`. Check the box when complete and add a `> ✅` note.

### M1 — Project Scaffold & Infra
- [x] M1.1 — Init repo, `pyproject.toml`, ruff + pytest, `.gitignore`, `.env.example`
> ✅ Done: git init + src/ package skeleton (ingest/graph/vector/retrieval/llm/eval), `pyproject.toml` (setuptools src-layout, ruff E/F/I, pytest pythonpath=src), `src/config.py` (dotenv-backed settings), `.env.example`, `.gitignore`, `tests/test_scaffold.py`. Verified: `ruff check .` clean, `pytest` 1 passed. Full `pip install -e ".[dev]"` (torch/ragas) left for user.
- [x] M1.2 — `docker-compose.yml` for Neo4j + Qdrant + Ollama; verify all containers come up healthy
> ✅ Done: `docker-compose.yml` with Neo4j 5 (+APOC, ports 7474/7687), Qdrant (6333/6334), Ollama (11434), named volumes, healthchecks (neo4j cypher-shell, ollama `ollama list`). Verified: `docker compose up -d` → neo4j & ollama `(healthy)`, qdrant `/readyz` → 200. Qdrant has no in-container healthcheck (image lacks curl/wget); checked from host.
- [x] M1.3 — Connection smoke tests (ping Neo4j, Qdrant, Ollama; pull one Ollama model)
> ✅ Done: `src/health.py` (lazy-import pings for all 3, `python src/health.py`, exit 0/1) + `tests/test_connections.py` (skips if a service is down). Pulled `qwen3:0.6b` (small smoke model; production model choice deferred to M8.3). Verified: all 3 OK, `pytest` 4 passed, ruff clean.
- [x] M1.4 — Write `README.md` skeleton + setup instructions
> ✅ Done: `README.md` with overview, architecture diagram, tech stack, prerequisites, 4-step setup (docker → venv → pull model → verify), config/layout/commands sections, and TBD Results/Demo placeholders. Includes the Python 3.11/3.12 venv note (torch/ragas wheels) as requested.

### M2 — Data Collection
- [x] M2.1 — Pick exact source (arXiv metadata snapshot / Kaggle / HF dataset) and document the choice
> ✅ Done: Chose **OpenAlex** (CC0, no API key, polite-pool `mailto`), focused NLP subfield. Documented in `docs/data-source.md` with reproducible filter `concepts.id:C204321447,from_publication_date:2024-01-01,cited_by_count:>10` (verified **5,847 works**), field→schema mapping, and two transforms (reconstruct inverted-index abstracts; keep in-corpus CITES for density). Verified live: API reachable, all required fields present.
- [x] M2.2 — Download script → `data/raw/`; cache so it isn't re-downloaded
> ✅ Done: `src/ingest/download.py` (stdlib urllib, cursor pagination, polite-pool mailto, 429/5xx retry/backoff, `--force`/`--max-records` flags) + `tests/test_download.py` (no-network). OpenAlex settings added to `config.py`/`.env.example`. Pulled **5,847 works** → `data/raw/openalex/works.jsonl` (55MB, gitignored) + `manifest.json`. Cache verified (re-run skips). Sample (Attention Is All You Need): 8 authors, 3 topics, 28 refs, abstract present.
- [x] M2.3 — Sanity-check schema: confirm fields for paper_id, title, authors, institution, topic, citations exist (or plan derivations)
> ✅ Done: `src/ingest/audit.py` + `tests/test_audit.py`. Coverage over 5,847 works: paper_id 100% (0 dupes), title 100%, year 100%, **abstract 77.2%** (use title when missing), authors 99.5% (25,988 distinct), authorships-with-institution 82.0% (4,389 institutions), topics 99.9% (993 distinct). **CITES is sparse**: 270,302 ref edges but only **1.2% (3,282) are in-corpus** (~26% papers cite in-corpus, ~28% cited). Derivations for M3: strip URL→short id; reconstruct inverted-index abstracts (fallback to title); normalize institution display_names. See Open Questions for the CITES-density decision (M4).

### M3 — Data Cleaning
- [x] M3.1 — Dedupe (`drop_duplicates`), handle missing values
> ✅ Done: `src/ingest/clean.py` + `tests/test_clean.py`. Parses raw Works → normalized paper records, dedupes by paper_id (`drop_duplicates`), reconstructs inverted-index abstracts (falls back to title), keeps **in-corpus CITES only** (decision (a)). Out → `data/interim/papers_clean.jsonl` (5,847 papers; 0 dupes, 0 missing title, 1,334 abstracts filled from title, 69 papers w/ no valid author, 3,232 in-corpus CITES edges). pytest 11 passed.
- [x] M3.2 — Normalize entity names (e.g. "Open AI"/"openai" → "OpenAI") with a canonical-name map
> ✅ Done: `src/ingest/normalize.py` + `tests/test_normalize.py`. Strips trailing `(Country)` from institution names (recognized-country list, so campus tags like `(Beijing)` survive), applies `CANONICAL_NAME_MAP` (open ai→OpenAI extension point), and **merges ids only for pure country-suffix splits** (Google 5→1, Microsoft 8→1; ambiguous same-name like Northeastern University kept separate). In→`papers_clean.jsonl`, out→`data/interim/papers_normalized.jsonl`. Stats: 2,085 suffixes stripped, 32 orgs merged (4,338→4,281 inst ids), 21 canonical-map hits. pytest 15 passed.
- [x] M3.3 — Emit `data/processed/clean_papers.csv` + a short data-quality report
> ✅ Done: `src/ingest/export.py` + `tests/test_export.py`. Flattens `papers_normalized.jsonl` → one-row-per-paper `data/processed/clean_papers.csv` (5,847 rows × 18 cols; nested authors/institutions/topics/refs joined with `" | "`, pandas-quoted so comma'd names round-trip) and writes `data/processed/data_quality_report.md`. Report highlights: title/year/cited_by 100%, real abstract 77.2%, ≥1 author 98.8%, ≥1 institution 86.7%, ≥1 topic 99.9%, ≥1 in-corpus cite 25.3%; 25,988 authors / 4,281 institutions / 993 topics / 3,232 CITES edges; years 2024–2026. pytest 18 passed, ruff clean. **M3 milestone complete.**

### M4 — Knowledge Graph (Neo4j)
- [x] M4.1 — Define node/edge schema + uniqueness constraints/indexes
> ✅ Done: `src/graph/schema.py` + `tests/test_schema.py`. Defines the model — nodes Paper(paper_id)/Author(author_id)/Institution(inst_id)/Topic(topic_id); edges (Paper)-[:AUTHORED_BY]->(Author), (Author)-[:WORKS_AT]->(Institution), (Paper)-[:STUDIES]->(Topic), (Paper)-[:CITES]->(Paper). Applies 4 idempotent uniqueness constraints (the MERGE keys for M4.2, each backed by a range index) + 6 property indexes (Paper.year/cited_by_count, Author/Institution/Topic.name, Topic.field) + a `paper_fulltext` index over title/abstract (NL→seed-node mapping for M6). CLI: `--apply` (default) / `--show` / `--drop` (schema-only, data untouched). Applied to live Neo4j and verified idempotent (re-run clean). pytest 5 passed (no-DB statement tests), ruff clean.
- [x] M4.2 — Ingestion script: rows → nodes (Paper/Author/Institution/Topic) + edges (AUTHORED_BY/WORKS_AT/STUDIES/CITES)
> ✅ Done: `src/graph/build.py` + `tests/test_build.py`. Ingests nested `papers_normalized.jsonl` (source of truth for author→institution nesting) via batched `UNWIND`+`MERGE` (idempotent); applies schema first, then pass 1 = nodes + AUTHORED_BY/WORKS_AT/STUDIES, pass 2 = CITES (after all Papers exist; dst MATCHed not MERGEd so no stray nodes). Flags: `--reset` (DETACH DELETE first), `--batch-size`. Built live: **Paper 5,847 / Author 25,988 / Institution 4,281 / Topic 993** (match M3.3 report) + **AUTHORED_BY 30,613 / WORKS_AT 28,739 / STUDIES 15,949 / CITES 3,232**. Re-run = identical counts (idempotent). Spot-check traversal (Attention Is All You Need): authors→orgs (Gomez→Toronto+Google) + topics correct. pytest 4 passed, ruff clean.
- [x] M4.3 — Verify counts (nodes, edges) and spot-check a few traversals in Neo4j Browser
> ✅ Done: `src/graph/verify.py` + `tests/test_verify.py`. Independently recomputes expected cardinalities from `papers_normalized.jsonl` (distinct-set MERGE semantics) and diffs vs. live graph — **all 8 match** (Paper 5,847 / Author 25,988 / Institution 4,281 / Topic 993 / AUTHORED_BY 30,613 / WORKS_AT 28,739 / STUDIES 15,949 / CITES 3,232). Integrity: 0 dup ids, 0 self-citations, 0 title-less papers; info-only gaps 69 papers w/o author (matches M3.1) + 5 w/o topic. Spot-checks: top institutions (Tsinghua 157), top in-corpus-cited papers, 3-hop Google→NLP (113 authors/19 papers). Exits 0 only if every count matches. Also prints copy-paste Cypher for Neo4j Browser (schema viz, transformer neighbourhood, topic→institutions, citation subgraph) for the M4.4 screenshots. pytest 3 passed, ruff clean. **M4.4 (screenshots) is a manual human step — queries provided.**
- [x] M4.4 — Capture 2–3 graph screenshots for the LinkedIn post
> ✅ Done: screenshots captured manually from Neo4j Browser (http://localhost:7474) using the Cypher printed by `src/graph/verify.py` (schema viz, transformer neighbourhood, topic→institutions, citation subgraph). **M4 milestone complete.**

### M5 — Vector RAG Baseline ✅
- [x] M5.1 — Chunk papers (title+abstract) and embed with bge-small
> ✅ Done: `src/vector/embed.py` + `tests/test_embed.py`. Pure word-window chunker (`title`\n\n`abstract`, max 256 words / 32 overlap; short abstracts → 1 chunk) + bge-small (`BAAI/bge-small-en-v1.5`, 384-dim) via sentence-transformers, L2-normalized so Qdrant cosine==dot (M5.2). Reads `papers_normalized.jsonl`, writes row-aligned `data/processed/chunks.jsonl` (chunk_id/paper_id/chunk_index + metadata + text) and `embeddings.npy`. Built live: **5,847 papers → 6,776 chunks** (929 from 200+-word multi-chunk papers) → vectors `(6776, 384)` float32, all norms 1.0, chunks↔vectors aligned 1:1. CLI: `--limit`/`--no-embed`/`--max-words`/`--overlap`/`--batch-size`. pytest 8 passed (chunking logic; embedding exercised at runtime), ruff clean.
- [x] M5.2 — Index into Qdrant; build similarity-search function
> ✅ Done: `src/vector/index.py` + `src/vector/search.py` + `tests/test_index.py`. `index.py` upserts the M5.1 chunks+vectors into a cosine collection (point id = row index, idempotent stable ids; `chunk_id`/`paper_id`/`text`+metadata in payload so search hands context straight to the LLM in M8); CLI `--reset`/`--batch-size`. `search.py` = `search(query, k)` that embeds the query with bge's retrieval prefix ("Represent this sentence for searching relevant passages: " — queries only; passages stay prefix-free), L2-normalized, then `query_points` top-k → payload+score; module-cached model, CLI `python src/vector/search.py --k N "..."`. Built live: **6,776 points** in collection `papers`; sanity query "transformer architecture based entirely on attention" → "Attention Is All You Need" + related transformer papers. pytest 4 passed (pure point-building + query-prefix; live Qdrant round-trip = idempotent re-index + nearest-neighbour, skips if Qdrant down) → 42 total, ruff clean.
- [x] M5.3 — Wrap as a baseline `vector_rag(query)` that returns context + LLM answer
> ✅ Done: `src/vector/rag.py` + `tests/test_rag.py`. `vector_rag(query, k)` = retrieve (M5.2 `search`) → grounded prompt (`SYSTEM_PROMPT`: "answer ONLY from the numbered passages, else say you don't know") → Ollama answer; returns `{query, answer, contexts}` (contexts = retrieved chunks+scores, so M9/M10 can show what the model saw). LLM call is a **minimal self-contained baseline** (`generate()` via `ollama.Client.chat`, temp 0, `<think>…</think>` reasoning-trace stripped) — the reusable wrapper + hardened template are M8, which replaces it and unifies both pipelines (M8.2). CLI `python src/vector/rag.py [--k N] "..."`. Aligned `config.OLLAMA_MODEL` default + `.env.example` to the pulled smoke model `qwen3:0.6b` (production pick deferred to M8.3) so the baseline runs out of the box. Verified end-to-end live (retrieve→answer→sources). pytest 7 passed (prompt assembly, think-strip, retrieve→prompt→generate wiring via monkeypatch — no live services) → 49 total, ruff clean.
- [x] M5.4 — Confirm baseline works end-to-end on a simple question
> ✅ Done: `tests/test_baseline_e2e.py`. Live run of `vector_rag` on a simple question — "Which neural network architecture is based solely on attention mechanisms, dispensing with recurrence and convolutions?" → retrieval ranks **"Attention Is All You Need" #1 (0.7263)** and the LLM answers correctly: *"The Transformer is based solely on attention mechanisms, dispensing with recurrence and convolutions."* The test asserts the expected paper is in the retrieved contexts (deterministic) + the answer is non-empty and on-topic (mentions transformer/attention, not "don't know"); skips cleanly if Qdrant/Ollama/index/model are absent. Note: meta-phrasing ("what does paper X propose?") retrieves worse than content-descriptive phrasing — a known bge-asymmetric trait and exactly the kind of multi-hop/relational gap Graph-RAG (M6) targets. pytest 1 passed (50 total, ruff clean). **M5 milestone complete.**

### M6 — Graph Retrieval Engine
- [ ] M6.1 — Build parametric Cypher templates for common multi-hop patterns (author→paper→topic→institution)
- [ ] M6.2 — `graph_retrieve(query)` that maps a question to a traversal and returns structured context
- [ ] M6.3 — Verify it answers a multi-hop question vector RAG can't

### M7 — Hybrid Retrieval + LangGraph Router
- [ ] M7.1 — Build LangGraph router: classify query as simple (→vector) vs relational/multi-hop (→graph), with hybrid fallback
- [ ] M7.2 — Context-merge step (combine graph + vector context cleanly, dedupe)
- [ ] M7.3 — End-to-end `answer(query)` pipeline wired through the router

### M8 — Local LLM Integration
- [ ] M8.1 — Ollama wrapper with a strict "answer ONLY from context" prompt template
- [ ] M8.2 — Wire LLM as the final answer-writer for both pipelines
- [ ] M8.3 — Pick the best small model after a quick quality/speed comparison

### M9 — Evaluation
- [ ] M9.1 — Author `eval/questions.jsonl`: 50 easy + 50 multi-hop with ground-truth answers
- [ ] M9.2 — Ragas harness comparing Vector-only vs Graph-RAG
- [ ] M9.3 — Produce results table (accuracy / precision / recall / faithfulness) + a chart
- [ ] M9.4 — Identify the single most dramatic multi-hop win for the demo

### M10 — Streamlit Demo
- [ ] M10.1 — UI: question box, answer panel, retrieved-graph panel, retrieved-docs panel
- [ ] M10.2 — Toggle between Vector RAG and Graph-RAG to show the contrast live
- [ ] M10.3 — Polish (loading states, example questions, basic styling)

### M11 — LinkedIn Showcase
- [ ] M11.1 — Screen recording of the side-by-side fail vs success
- [ ] M11.2 — Architecture diagram (Neo4j → LangGraph → Ollama → Answer)
- [ ] M11.3 — Draft the post copy + finalize README with results + GIFs

---

## 5. Pre-Build Artifacts (Do First, Once)
- [x] SDD — Short Software Design Doc (1–2 pages): problem, architecture, data flow, decisions
> ✅ Skipped by decision (2026-06-09): CLAUDE.md serves as the design doc for v1.
- [x] This CLAUDE.md reviewed and agreed
> ✅ Reviewed/agreed 2026-06-09; tech stack (§2) and CITES decision (§10) confirmed by user.
- [x] Definition of Done agreed (see Section 6)
> ✅ Agreed 2026-06-09 (Section 6 unchanged).

---

## 6. Definition of Done (v1)
- Both pipelines run locally with zero paid API calls.
- Graph-RAG measurably beats vector RAG on the multi-hop split (numbers in README).
- Streamlit demo runs from a single command after `docker compose up`.
- README has setup, results table, and demo GIF.
- At least one multi-hop question where vector RAG visibly fails and Graph-RAG succeeds.

---

## 7. Guardrails & Conventions
- **Reproducibility:** every script runnable from CLI; seeds fixed where relevant.
- **Config over hardcoding:** connection strings, model names, paths in `.env` / a config module.
- **Tests:** at least smoke tests for ingest, graph query, vector search, router.
- **Commits:** one logical commit per completed sub-task; message references the task ID (e.g. `M4.2: neo4j ingestion`).
- **No scope creep:** if a new idea appears, log it in Section 8, don't build it.

---

## 8. Backlog / v2 (Do NOT build in v1)
- [ ] GNN layer (PyTorch Geometric, GCN/GAT) for learned graph embeddings + link prediction
- [ ] MCP server (TypeScript/Python) exposing the engine to Claude Desktop / Cursor
- [ ] Larger dataset / streaming ingestion
- [ ] Better entity resolution (fuzzy matching, embeddings-based dedupe)

---

## 9. Status Log
> Append newest entries at the top. Format: `YYYY-MM-DD — what changed — next up`.

- 2026-06-09 — M5.4 done: tests/test_baseline_e2e.py runs vector_rag live on a simple question → "Attention Is All You Need" retrieved #1, LLM answers correctly ("The Transformer is based solely on attention..."); skips if stack/index/model absent. pytest 1 passed (50 total). **M5 milestone complete.** — Next: M6.1 (parametric Cypher templates for multi-hop patterns: author→paper→topic→institution).
- 2026-06-09 — M5.3 done: src/vector/rag.py vector_rag(query,k) = retrieve→grounded prompt→Ollama answer, returns {query,answer,contexts}; minimal baseline LLM call (M8 will unify). Aligned OLLAMA_MODEL default to pulled qwen3:0.6b. Verified end-to-end live. pytest 7 passed (49 total). — Next: M5.4 (confirm baseline works end-to-end on a simple question).
- 2026-06-09 — M5.2 done: src/vector/index.py upserts 6,776 chunks+vectors into Qdrant cosine collection 'papers' (idempotent row-index ids, payload carries text); src/vector/search.py = search(query,k) with bge query prefix → query_points top-k. Sanity query returns "Attention Is All You Need". pytest 4 passed (42 total). — Next: M5.3 (wrap as vector_rag(query) returning context + LLM answer).
- 2026-06-09 — M5.1 done: src/vector/embed.py chunks title+abstract (word-window, 256/32) and embeds with bge-small (384-dim, L2-normalized) → data/processed/chunks.jsonl + embeddings.npy. Built: 5,847 papers → 6,776 chunks → vectors (6776, 384), aligned 1:1. pytest 8 passed (38 total). — Next: M5.2 (index chunks+vectors into Qdrant; build similarity-search function).
- 2026-06-09 — M4.4 done (manual): captured Neo4j Browser screenshots for the LinkedIn post using verify.py's Cypher. **M4 milestone complete.** — Next: M5.1 (chunk papers title+abstract, embed with bge-small).
- 2026-06-09 — M4.3 done: src/graph/verify.py independently recomputes expected counts from JSONL and diffs vs. live graph — all 8 node/edge counts match; integrity clean (0 dup/self-cite/title-less); spot-check traversals OK; prints Browser Cypher for M4.4. pytest 3 passed. — Next: M4.4 (manual: capture 2–3 Neo4j Browser screenshots for the LinkedIn post), then M5.1 (chunk+embed papers with bge-small).
- 2026-06-09 — M4.2 done: src/graph/build.py ingests papers_normalized.jsonl → Neo4j (batched UNWIND+MERGE, idempotent). Counts: 5,847 papers / 25,988 authors / 4,281 insts / 993 topics; 30,613 AUTHORED_BY / 28,739 WORKS_AT / 15,949 STUDIES / 3,232 CITES. pytest 4 passed. — Next: M4.3 (verify counts + spot-check traversals in Neo4j Browser).
- 2026-06-09 — M4.1 done: src/graph/schema.py defines node/edge model + applies 4 uniqueness constraints + 6 property indexes + 1 fulltext index (idempotent, CLI --apply/--show/--drop). Applied to live Neo4j; pytest 5 passed. — Next: M4.2 (ingestion: rows → Paper/Author/Institution/Topic nodes + AUTHORED_BY/WORKS_AT/STUDIES/CITES edges).
- 2026-06-09 — M3.3 done: src/ingest/export.py flattens papers_normalized.jsonl → data/processed/clean_papers.csv (5,847×18) + data_quality_report.md. pytest 18 passed. **M3 milestone complete.** — Next: M4.1 (Neo4j node/edge schema + uniqueness constraints/indexes).
- 2026-06-09 — M3.2 done: src/ingest/normalize.py (country-suffix strip + canonical map + safe country-split merge) → data/interim/papers_normalized.jsonl. 32 orgs merged, ambiguous names preserved. pytest 15 passed. — Next: M3.3 (emit data/processed/clean_papers.csv + data-quality report).
- 2026-06-09 — M3.1 done: src/ingest/clean.py dedupe + missing-value handling → data/interim/papers_clean.jsonl (5,847). CITES decision (a) adopted (in-corpus only). pytest 11 passed. — Next: M3.2 (normalize entity names via canonical-name map).
- 2026-06-09 — M2.3 done: src/ingest/audit.py coverage report. All core fields well-covered EXCEPT in-corpus citations (1.2%, 3,282 edges) — flagged as Open Question for M4. pytest 8 passed. **M2 milestone complete.** — Next: M3.1 (dedupe + missing-value handling).
- 2026-06-09 — M2.2 done: src/ingest/download.py (cached, cursor-paginated, retry) pulled 5,847 OpenAlex works → data/raw/openalex/works.jsonl (55MB) + manifest. pytest 6 passed. — Next: M2.3 (sanity-check schema: confirm/plan fields for paper/author/institution/topic/citations).
- 2026-06-09 — M2.1 done: data source = OpenAlex, focused NLP subset (~5.8k), documented in docs/data-source.md; API + fields verified live. — Next: M2.2 (download script → data/raw/, cached).
- 2026-06-09 — M1.4 done: README.md (overview, architecture, setup, layout, commands, Python 3.11/3.12 venv note). **M1 milestone complete.** — Next: M2.1 (pick + document the dataset source).
- 2026-06-09 — M1.3 done: src/health.py + tests/test_connections.py (all 3 services ping OK), pulled qwen3:0.6b smoke model. pytest 4 passed. — Next: M1.4 (README skeleton + setup instructions; include Python 3.11/3.12 venv note).
- 2026-06-09 — M1.2 done: docker-compose (Neo4j+APOC, Qdrant, Ollama) with healthchecks + named volumes; all containers verified healthy (qdrant /readyz=200). — Next: M1.3 (connection smoke tests + pull an Ollama model).
- 2026-06-09 — M1.1 done: repo scaffold (src/ skeleton, pyproject, ruff+pytest, config.py, .env.example, .gitignore). SDD skipped (CLAUDE.md serves as design doc). ruff/pytest green. — Next: M1.2 (docker-compose: Neo4j + Qdrant + Ollama).
- 2026-06-08 — CLAUDE.md created. No code yet. — Next: SDD + M1.1.

---

## 10. Open Questions
> Claude logs blockers/decisions needing the user here.

- **[RESOLVED 2026-06-09 → (a)] CITES density is low.** Only 1.2% of references point within the corpus. **Decision: keep in-corpus CITES only, no backfill** (user-approved). Implemented in M3.1 (`clean.py` filters references to the corpus id set; 3,232 edges after removing self/dupe). Multi-hop will lean on author/topic/institution relationships (82–99% coverage).
