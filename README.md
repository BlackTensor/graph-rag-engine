# Graph-RAG Discovery Engine

A **Graph-RAG** system over the **AI Research Papers** domain that beats traditional
vector RAG on **multi-hop reasoning** questions — running **fully local** (no API costs).

> **The money shot:** a side-by-side where vector RAG *fails* a multi-hop question
> and Graph-RAG *nails* it.

- **Domain:** AI research papers
- **Entities:** `Paper`, `Author`, `Institution`, `Topic`
- **Relationships:** `AUTHORED_BY`, `WORKS_AT`, `STUDIES`, `CITES`
- **Scope (v1):** no GNN — Neo4j native graph traversal handles multi-hop. (GNN is a v2 stretch goal.)

> 🚧 **Status:** under active development. See [`CLAUDE.md`](CLAUDE.md) for the live
> milestone checklist and progress log.

---

## Architecture

```
              ┌──────────────┐
   question → │  LangGraph   │  classify: simple → vector | relational → graph
              │   router     │
              └──────┬───────┘
            ┌────────┴────────┐
            ▼                 ▼
   ┌────────────────┐  ┌──────────────┐
   │  Vector RAG    │  │  Graph RAG   │
   │  Qdrant +      │  │  Neo4j       │
   │  bge-small     │  │  Cypher      │
   └───────┬────────┘  └──────┬───────┘
           └────────┬─────────┘
                    ▼
            ┌──────────────┐
            │  Ollama LLM  │ → answer (strictly from retrieved context)
            └──────────────┘
```

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Graph DB | Neo4j (Docker) + `neo4j` driver |
| Vector DB | Qdrant (Docker) + `qdrant-client` |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Orchestration | LangGraph (query router + hybrid retrieval) |
| Local LLM | Ollama (`qwen3` / `gemma`) |
| Evaluation | Ragas |
| Demo UI | Streamlit |

---

## Prerequisites

- **Docker** + Docker Compose (for Neo4j, Qdrant, Ollama)
- **Python 3.11 or 3.12** — see the venv note below
- ~6 GB free disk for container images + at least one Ollama model

> ⚠️ **Use Python 3.11 or 3.12.** Some runtime dependencies (PyTorch via
> `sentence-transformers`, and `ragas`) may not yet publish wheels for the newest
> Python (e.g. 3.13/3.14), which makes `pip install` fail or fall back to slow
> source builds. Create the virtual environment with a 3.11/3.12 interpreter.

---

## Setup

### 1. Start the infrastructure

```bash
cp .env.example .env          # adjust passwords/ports if you like
docker compose up -d          # Neo4j + Qdrant + Ollama
docker compose ps             # confirm neo4j & ollama are (healthy)
```

Service endpoints:

| Service | URL | Notes |
|---|---|---|
| Neo4j Browser | http://localhost:7474 | user `neo4j`, pwd from `.env` |
| Neo4j Bolt | `bolt://localhost:7687` | driver connections |
| Qdrant | http://localhost:6333 | REST + dashboard |
| Ollama | http://localhost:11434 | LLM API |

### 2. Create the Python environment

```bash
# Use a 3.11/3.12 interpreter (see the note above)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # runtime + dev (ruff, pytest)
```

### 3. Pull an Ollama model

```bash
docker exec graphrag-ollama ollama pull qwen3:0.6b   # small smoke model
# the production model is chosen later (see CLAUDE.md M8.3)
```

### 4. Verify everything is wired up

```bash
python src/health.py    # pings Neo4j, Qdrant, Ollama; exits 0 if all OK
pytest -q               # smoke + connection tests
ruff check .            # lint
```

---

## Configuration

All connection strings, model names, and paths live in `.env` (template in
`.env.example`) and are read centrally by [`src/config.py`](src/config.py). Import
settings from there — don't hardcode them elsewhere.

## Repository Layout

```
.
├── CLAUDE.md              # project memory + milestone checklist (source of truth)
├── docker-compose.yml     # Neo4j + Qdrant + Ollama
├── pyproject.toml
├── .env.example
├── data/                  # raw / interim / processed datasets
├── src/
│   ├── config.py          # central settings
│   ├── health.py          # connectivity smoke checks
│   ├── ingest/            # download + clean
│   ├── graph/             # neo4j build + cypher queries
│   ├── vector/            # qdrant index + search
│   ├── retrieval/         # hybrid + langgraph router
│   ├── llm/               # ollama wrapper
│   └── eval/              # ragas harness
├── eval/                  # questions.jsonl (50 easy + 50 multi-hop)
├── app/                   # streamlit_app.py
└── tests/
```

## Common Commands

```bash
docker compose up -d        # start stack
docker compose ps           # health
docker compose down         # stop (add -v to wipe volumes/data)
python src/health.py        # connectivity check
pytest -q                   # tests
ruff check .                # lint
```

---

## Results

_TBD — populated in M9 (evaluation) and M11 (showcase): Vector-only vs Graph-RAG
accuracy / precision / recall / faithfulness, plus the headline multi-hop win._

## Demo

_TBD — Streamlit side-by-side (M10) and a recorded GIF (M11)._
