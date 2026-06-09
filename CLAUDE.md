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

## 3. Repository Layout (Target)

```
graphrag-discovery/
├── CLAUDE.md                 # this file
├── README.md
├── docker-compose.yml        # neo4j + qdrant + ollama
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/                  # downloaded datasets
│   ├── interim/
│   └── processed/            # clean_papers.csv
├── src/
│   ├── ingest/               # download + clean
│   ├── graph/                # neo4j build + queries
│   ├── vector/               # qdrant index + search
│   ├── retrieval/            # hybrid + router (langgraph)
│   ├── llm/                  # ollama wrapper
│   └── eval/                 # ragas harness
├── eval/
│   └── questions.jsonl       # 50 easy + 50 multi-hop
├── app/
│   └── streamlit_app.py
└── tests/
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
- [ ] M1.4 — Write `README.md` skeleton + setup instructions
> 📝 Reminder: README must note that runtime deps (torch via sentence-transformers, ragas) may lack wheels on Python 3.14 — recommend a Python 3.11/3.12 venv for `pip install -e ".[dev]"`.

### M2 — Data Collection
- [ ] M2.1 — Pick exact source (arXiv metadata snapshot / Kaggle / HF dataset) and document the choice
- [ ] M2.2 — Download script → `data/raw/`; cache so it isn't re-downloaded
- [ ] M2.3 — Sanity-check schema: confirm fields for paper_id, title, authors, institution, topic, citations exist (or plan derivations)

### M3 — Data Cleaning
- [ ] M3.1 — Dedupe (`drop_duplicates`), handle missing values
- [ ] M3.2 — Normalize entity names (e.g. "Open AI"/"openai" → "OpenAI") with a canonical-name map
- [ ] M3.3 — Emit `data/processed/clean_papers.csv` + a short data-quality report

### M4 — Knowledge Graph (Neo4j)
- [ ] M4.1 — Define node/edge schema + uniqueness constraints/indexes
- [ ] M4.2 — Ingestion script: rows → nodes (Paper/Author/Institution/Topic) + edges (AUTHORED_BY/WORKS_AT/STUDIES/CITES)
- [ ] M4.3 — Verify counts (nodes, edges) and spot-check a few traversals in Neo4j Browser
- [ ] M4.4 — Capture 2–3 graph screenshots for the LinkedIn post

### M5 — Vector RAG Baseline
- [ ] M5.1 — Chunk papers (title+abstract) and embed with bge-small
- [ ] M5.2 — Index into Qdrant; build similarity-search function
- [ ] M5.3 — Wrap as a baseline `vector_rag(query)` that returns context + LLM answer
- [ ] M5.4 — Confirm baseline works end-to-end on a simple question

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
- [ ] SDD — Short Software Design Doc (1–2 pages): problem, architecture, data flow, decisions
- [ ] This CLAUDE.md reviewed and agreed
- [ ] Definition of Done agreed (see Section 6)

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

- 2026-06-09 — M1.3 done: src/health.py + tests/test_connections.py (all 3 services ping OK), pulled qwen3:0.6b smoke model. pytest 4 passed. — Next: M1.4 (README skeleton + setup instructions; include Python 3.11/3.12 venv note).
- 2026-06-09 — M1.2 done: docker-compose (Neo4j+APOC, Qdrant, Ollama) with healthchecks + named volumes; all containers verified healthy (qdrant /readyz=200). — Next: M1.3 (connection smoke tests + pull an Ollama model).
- 2026-06-09 — M1.1 done: repo scaffold (src/ skeleton, pyproject, ruff+pytest, config.py, .env.example, .gitignore). SDD skipped (CLAUDE.md serves as design doc). ruff/pytest green. — Next: M1.2 (docker-compose: Neo4j + Qdrant + Ollama).
- 2026-06-08 — CLAUDE.md created. No code yet. — Next: SDD + M1.1.

---

## 10. Open Questions
> Claude logs blockers/decisions needing the user here.

- (none yet)
