"""Tests for the graph retriever (M6.2).

Pure logic (template selection, topic token-overlap, seed-type priority, the
abstain path) runs everywhere via a fake session; the end-to-end traversal needs
a live Neo4j and is SKIPPED when unreachable (mirrors tests/test_index.py).
"""

import pytest

import config
from graph import retrieve


class _FakeSession:
    """Replays canned rows per Cypher fragment so resolution can be driven."""

    def __init__(self, authors=None, institutions=None, topics=None):
        self._authors = authors or []
        self._institutions = institutions or []
        self._topics = topics or []

    def run(self, cypher, **params):
        if ":Author" in cypher:
            q = params["q"].lower()
            return iter(
                [{"name": n} for n in self._authors if n.lower() in q]
            )
        if ":Institution" in cypher:
            q = params["q"].lower()
            return iter(
                [{"name": n} for n in self._institutions if n.lower() in q]
            )
        if ":Topic" in cypher and "RETURN t.name" in cypher:
            return iter([{"name": n} for n in self._topics])
        # Fallback: a template body run by run_template -> return empty rows.
        return iter([])


# --- select_template (pure) ---------------------------------------------------

@pytest.mark.parametrize(
    "seed_type,query,expected",
    [
        ("author", "What topics does Yoshua Bengio work on?", "author_topics"),
        ("author", "Where does Yoshua Bengio work?", "author_institutions"),
        ("author", "Who collaborates with Yoshua Bengio?", "coauthors"),
        ("author", "Which institutions work on the same topics as Bengio?",
         "peer_institutions_via_topics"),
        ("author", "Whose work does Bengio cite?", "author_cited_institutions"),
        ("institution", "Who publishes at Google?", "institution_authors"),
        ("institution", "What does Google research?", "institution_topics"),
        ("institution", "Which orgs collaborate with Google?",
         "collaborating_institutions"),
        ("topic", "Which institutions lead on this topic?", "topic_institutions"),
        ("topic", "Who are the leading authors here?", "topic_authors"),
        ("topic", "What are the most influential papers?", "topic_top_cited_papers"),
    ],
)
def test_select_template(seed_type, query, expected):
    assert retrieve.select_template(seed_type, query) == expected


def test_select_template_unknown_seed_is_none():
    assert retrieve.select_template("paper", "anything") is None


def test_selected_templates_all_exist():
    # Every name select_template can emit must be a real M6.1 template.
    for seed in ("author", "institution", "topic"):
        name = retrieve.select_template(seed, "what does it study and cite")
        assert name in retrieve.templates.TEMPLATES


# --- topic token overlap (pure) ----------------------------------------------

def test_best_topic_matches_on_token_overlap():
    names = ["Topic Modeling", "Natural Language Processing Techniques", "Computer Vision"]
    assert (
        retrieve._best_topic("who works on natural language processing?", names)
        == "Natural Language Processing Techniques"
    )
    assert retrieve._best_topic("topic modeling research", names) == "Topic Modeling"


def test_best_topic_below_threshold_returns_none():
    names = ["Natural Language Processing Techniques"]
    assert retrieve._best_topic("tell me about reinforcement learning", names) is None


# --- seed priority + resolution (fake session) -------------------------------

def test_resolve_seed_prefers_author_over_institution_and_topic():
    sess = _FakeSession(
        authors=["Ashish Vaswani"], institutions=["Google"], topics=["Topic Modeling"]
    )
    assert retrieve.resolve_seed(sess, "Where does Ashish Vaswani at Google work?") == (
        "author", "Ashish Vaswani",
    )


def test_resolve_seed_institution_when_no_author():
    sess = _FakeSession(institutions=["Google"], topics=["Topic Modeling"])
    assert retrieve.resolve_seed(sess, "Who publishes at Google?") == (
        "institution", "Google",
    )


def test_resolve_seed_topic_fallback():
    sess = _FakeSession(topics=["Natural Language Processing Techniques"])
    seed_type, name = retrieve.resolve_seed(
        sess, "which institutions lead on natural language processing?"
    )
    assert seed_type == "topic"
    assert name == "Natural Language Processing Techniques"


def test_graph_retrieve_abstains_when_no_entity():
    sess = _FakeSession()  # nothing resolves
    result = retrieve.graph_retrieve("what is the meaning of life?", session=sess)
    assert result["template"] is None
    assert result["rows"] == []
    assert "No graph traversal" in result["context"]


# --- rendering ----------------------------------------------------------------

def test_render_context_lists_rows():
    out = retrieve.render_context(
        "topic_institutions", "topic", "NLP",
        [{"institution": "Google", "papers": 12}],
    )
    assert "topic_institutions" in out and "Google" in out and "papers=12" in out


def test_render_context_handles_no_rows():
    out = retrieve.render_context("author_topics", "author", "X", [])
    assert "no matching paths" in out


# --- live end-to-end (skips without Neo4j) -----------------------------------

def _neo4j_or_skip():
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        return driver
    except Exception as exc:  # noqa: BLE001 - integration test, skip if down
        pytest.skip(f"Neo4j unreachable ({config.NEO4J_URI}): {exc}")


def test_graph_retrieve_live_multihop():
    driver = _neo4j_or_skip()
    try:
        with driver.session() as session:
            result = retrieve.graph_retrieve(
                "Which institutions work on the same topics as Ashish Vaswani?",
                session=session,
                limit=5,
            )
    finally:
        driver.close()
    # Author seed resolved and routed to the 4-hop chain.
    assert result["seed_type"] == "author"
    assert result["template"] == "peer_institutions_via_topics"
    assert result["rows"], "expected at least one peer institution"
    assert "institution" in result["rows"][0]
