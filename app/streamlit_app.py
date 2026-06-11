"""Streamlit demo — Graph-RAG Discovery Engine (M10.1/M10.2/M10.3).

    cd "RAG Graph P"
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Graph-RAG Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light CSS: tighten top padding; style the radio selector.
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stRadio"] label p { font-weight: 600; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached module loaders (one build per session) ─────────────────────────────

@st.cache_resource(show_spinner="Loading retrieval pipeline…")
def load_pipeline():
    from retrieval import pipeline  # noqa: PLC0415
    return pipeline


@st.cache_resource(show_spinner="Loading vector RAG…")
def load_vector_rag():
    from vector import rag  # noqa: PLC0415
    return rag


@st.cache_resource(show_spinner="Loading graph compare…")
def load_graph_compare():
    from graph import compare  # noqa: PLC0415
    return compare


# ── Constants ─────────────────────────────────────────────────────────────────

_DEMO_Q = "Which institutions collaborate most with Tsinghua University?"

# Gold answer entities for the demo question (M9.4 benchmark).
_DEMO_GOLD = [
    "University of Hong Kong",
    "Chinese University of Hong Kong",
    "Shanghai Artificial Intelligence Laboratory",
    "Hong Kong University of Science and Technology",
    "Beijing Academy of Artificial Intelligence",
]

# Canonical LLM refusal string (from llm/client.py).
_INSUFFICIENT = "I don't know based on the available context."

_EXAMPLES: list[tuple[str, str]] = [
    ("multi-hop", _DEMO_Q),
    ("multi-hop", "What topics does Tsinghua University publish research on?"),
    ("multi-hop", "Which institutions work on the same topics as Ashish Vaswani?"),
    ("multi-hop", "Who collaborates with Microsoft Research?"),
    ("multi-hop", "Which researchers work at Tsinghua University?"),
    ("easy", "What is the Transformer architecture?"),
    ("easy", "What is BERT and how does it work?"),
    ("easy", "How does contrastive learning work?"),
]

_ROUTE_COLOR = {"graph": "green", "vector": "blue", "hybrid": "orange"}

_STATS = [
    ("Papers", "5,847"),
    ("Authors", "25,988"),
    ("Institutions", "4,281"),
    ("Citation edges", "3,232"),
]

# ── Session-state init ────────────────────────────────────────────────────────

for _k, _v in [
    ("last_query", ""),
    ("mode", "Smart (auto-route)"),
    ("result", None),
    ("compare_result", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_insufficient(answer: str) -> bool:
    return _INSUFFICIENT.lower() in answer.lower()


def _entity_recall(answer: str, gold: list[str]) -> tuple[int, int]:
    found = sum(1 for e in gold if e.lower() in answer.lower())
    return found, len(gold)


def _render_passages(contexts: list[dict], heading: str = "Retrieved Documents") -> None:
    st.markdown(f"##### {heading}")
    if not contexts:
        st.info("No document passages retrieved.")
        return
    for i, c in enumerate(contexts, 1):
        title = (c.get("title") or "(untitled)").strip()
        year = c.get("year", "")
        cites = c.get("cited_by_count")
        score = c.get("score")
        text = (c.get("text") or "").strip()
        if title and text.startswith(title):
            text = text[len(title):].lstrip("\n").strip()
        meta: list[str] = []
        if year:
            meta.append(str(year))
        if cites is not None:
            meta.append(f"{cites} cites")
        if score is not None:
            meta.append(f"score {score:.4f}")
        with st.expander(f"[{i}] {title}"):
            if meta:
                st.caption("  ·  ".join(meta))
            if text:
                st.write(text[:600] + ("…" if len(text) > 600 else ""))


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _pick(q: str) -> None:
    st.session_state["q_input"] = q
    st.session_state["last_query"] = q
    st.session_state["result"] = None
    st.session_state["compare_result"] = None
    st.rerun()


with st.sidebar:
    st.header("Example Questions")

    st.caption("**Multi-hop** — Graph-RAG shines")
    for kind, q in _EXAMPLES:
        if kind != "multi-hop":
            continue
        label = ("⭐ " if q == _DEMO_Q else "") + q
        if st.button(label, key=f"ex_{q}", use_container_width=True):
            _pick(q)

    st.caption("**Definitional** — Vector RAG")
    for kind, q in _EXAMPLES:
        if kind != "easy":
            continue
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            _pick(q)

    st.divider()
    st.caption("Architected & built by **Shayan Ansari** © 2026")
    st.caption(
        "**Stack**  \nNeo4j · Qdrant · Ollama · LangGraph  \n\n"
        "**Model**  \nqwen3 (local, fully offline)  \n\n"
        "**Data**  \n5,847 NLP papers (OpenAlex)"
    )

# ── Header + stats ────────────────────────────────────────────────────────────

st.title("Graph-RAG Discovery Engine")
st.caption(
    "AI Research Papers knowledge base · "
    "multi-hop relational questions → Graph-RAG · "
    "definitional questions → Vector RAG"
)

stat_cols = st.columns(len(_STATS))
for col, (label, val) in zip(stat_cols, _STATS):
    col.metric(label, val)

st.divider()

# ── Mode selector ─────────────────────────────────────────────────────────────

mode = st.radio(
    "mode",
    ["Smart (auto-route)", "Compare (side-by-side)"],
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state["mode"] = mode

# ── Question form ─────────────────────────────────────────────────────────────

with st.form("q_form", clear_on_submit=False):
    query_input = st.text_input(
        "Question",
        key="q_input",
        placeholder=_DEMO_Q,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", type="primary")

# Warm-up hint shown only before any result is cached for the current mode.
_has_result = (
    st.session_state["result"] is not None
    if mode == "Smart (auto-route)"
    else st.session_state["compare_result"] is not None
)
if st.session_state["last_query"] and not _has_result:
    st.caption("Running local LLM (qwen3) — first call warms up the model (~15–30 s).")
elif not st.session_state["last_query"]:
    st.caption("Runs fully offline — Neo4j · Qdrant · Ollama, no external API calls.")

if submitted and query_input.strip():
    q = query_input.strip()
    if q != st.session_state["last_query"]:
        st.session_state["last_query"] = q
        st.session_state["result"] = None
        st.session_state["compare_result"] = None

# ── Pipeline execution ────────────────────────────────────────────────────────

_query = st.session_state["last_query"]

if mode == "Smart (auto-route)":
    if _query and st.session_state["result"] is None:
        with st.spinner("Retrieving context and generating answer…"):
            try:
                st.session_state["result"] = load_pipeline().answer(_query)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

else:
    if _query and st.session_state["compare_result"] is None:
        with st.spinner("Vector RAG: embedding query and retrieving chunks… (step 1 of 2)"):
            try:
                _v = load_vector_rag().vector_rag(_query, k=5)
            except Exception as exc:
                st.error(f"Vector RAG error: {exc}")
                st.stop()
        with st.spinner("Graph-RAG: resolving entity, traversing graph… (step 2 of 2)"):
            try:
                _g = load_graph_compare().graph_rag(_query, limit=15)
            except Exception as exc:
                st.error(f"Graph-RAG error: {exc}")
                st.stop()
        st.session_state["compare_result"] = {
            "query": _query,
            "vector": _v,
            "graph": _g,
        }

# ── Empty state ───────────────────────────────────────────────────────────────

if not _query:
    st.markdown("#### How it works")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**Vector RAG** (semantic search)")
            st.markdown(
                "Embeds your question with *bge-small* and retrieves the nearest "
                "title + abstract chunks from Qdrant.  \n\n"
                "Best for: *What is X? How does Y work? Explain Z.*"
            )
    with c2:
        with st.container(border=True):
            st.markdown("**Graph-RAG** (multi-hop traversal)")
            st.markdown(
                "Maps your question to a named entity in the Neo4j knowledge graph "
                "and walks AUTHORED\\_BY / WORKS\\_AT / STUDIES / CITES edges.  \n\n"
                "Best for: *Who collaborates with X? Which institutions work on Y?*"
            )
    st.markdown("")
    st.markdown("**Try the demo question** (shows where Vector RAG fails):")
    if st.button(f"⭐  {_DEMO_Q}", type="primary", use_container_width=False):
        _pick(_DEMO_Q)
    st.stop()

# ── Display — Smart mode ──────────────────────────────────────────────────────

if mode == "Smart (auto-route)":
    result = st.session_state.get("result")
    if result is None:
        st.stop()

    route = result.get("route", "vector")
    reason = result.get("reason", "")
    executed = result.get("executed") or []
    fellback = result.get("fellback", False)
    answer_text = result.get("answer", "")
    passages = result.get("passages") or []
    graph_result = result.get("graph_result") or {}

    color = _ROUTE_COLOR.get(route, "gray")
    parts: list[str] = [f":{color}[**{route.upper()}**]", reason]
    if executed:
        parts.append("executed: " + " → ".join(f"`{e}`" for e in executed))
    if fellback:
        parts.append("⚠️ graph abstained → fell back to vector")
    st.markdown("  ·  ".join(parts))

    st.markdown("### Answer")
    with st.container(border=True):
        if _is_insufficient(answer_text):
            st.warning(answer_text)
        else:
            st.markdown(answer_text)

    st.divider()
    col_graph, col_docs = st.columns(2, gap="medium")

    with col_graph:
        st.markdown("#### Knowledge Graph")
        if graph_result.get("template"):
            st.caption(
                f"Template: `{graph_result['template']}`  ·  "
                f"seed {graph_result.get('seed_type')} = **{graph_result.get('seed_name')}**"
            )
            rows = graph_result.get("rows") or []
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Graph traversal returned no rows.")
        else:
            st.info("No graph traversal for this question.")

    with col_docs:
        st.markdown("#### Retrieved Documents")
        _render_passages(passages)

# ── Display — Compare mode ────────────────────────────────────────────────────

else:
    cr = st.session_state.get("compare_result")
    if cr is None:
        st.stop()

    v = cr["vector"]
    g = cr["graph"]
    graph_on = bool(g.get("template"))
    is_demo = _DEMO_Q.lower() in _query.lower()

    col_v, col_g = st.columns(2, gap="large")

    # ── Vector RAG column ─────────────────────────────────────────────────────
    with col_v:
        st.markdown("### :blue[Vector RAG]")
        st.caption(
            "Searches title + abstract chunks only — "
            "author/institution relationships are **not** stored in the text index."
            if graph_on
            else "Semantic search over title + abstract chunks."
        )

        v_answer = v.get("answer", "")
        v_insuff = _is_insufficient(v_answer)

        if v_insuff:
            st.error("Could not answer from retrieved passages")
        elif graph_on:
            st.warning("Answer from text chunks — may miss relational facts")
        else:
            st.success("Answered")

        with st.container(border=True):
            st.markdown(v_answer)

        # Entity recall for demo question.
        if is_demo:
            found, total = _entity_recall(v_answer, _DEMO_GOLD)
            st.metric(
                "Gold entities in answer",
                f"{found} / {total}",
                delta=f"{found - total} vs perfect" if found < total else "all found",
                delta_color="inverse" if found < total else "normal",
            )

        st.divider()
        _render_passages(v.get("contexts") or [], heading="Retrieved Chunks")

    # ── Graph-RAG column ──────────────────────────────────────────────────────
    with col_g:
        if graph_on:
            st.markdown("### :green[Graph-RAG]")
            st.caption(
                f"Graph traversal `{g['template']}` · "
                f"seed {g.get('seed_type')} = **{g.get('seed_name')}**"
            )
        else:
            st.markdown("### :orange[Graph-RAG (abstained)]")
            st.caption(
                "No entity recognized — graph could not run a traversal for this question."
            )

        g_answer = g.get("answer", "")
        g_insuff = _is_insufficient(g_answer)

        if graph_on and not g_insuff:
            st.success("Answered from knowledge graph")
        elif graph_on and g_insuff:
            st.warning("Graph resolved an entity but could not answer")
        else:
            st.error("Graph abstained — fell back to 'I don't know'")

        with st.container(border=True):
            st.markdown(g_answer)

        if is_demo:
            found, total = _entity_recall(g_answer, _DEMO_GOLD)
            st.metric(
                "Gold entities in answer",
                f"{found} / {total}",
                delta=f"+{found}" if found > 0 else "none found",
                delta_color="normal" if found > 0 else "inverse",
            )

        st.divider()
        if graph_on:
            st.markdown("##### Traversal Rows")
            rows = g.get("rows") or []
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Graph traversal returned no rows.")
        else:
            st.info(
                "Graph abstained — no Author, Institution, or Topic matched.  \n"
                "Try naming a specific institution or researcher."
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Graph-RAG Discovery Engine · Architected & built by Shayan Ansari © 2026 · "
    "5,847 NLP papers (OpenAlex CC0) · Neo4j 5 · Qdrant · qwen3 via Ollama · "
    "LangGraph · fully local, zero API cost"
)
