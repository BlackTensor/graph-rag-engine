"""Unit tests for the parametric Cypher templates (M6.1).

No live Neo4j: covers template well-formedness (declared params match the
placeholders in the Cypher, every template is LIMIT-bounded) and the
`run_template` param-validation contract. A tiny fake session captures the
Cypher + params that would be sent so the runner is exercised without a DB.
"""

import pytest

from graph import templates
from graph.templates import QueryTemplate, render_rows, run_template


class _FakeSession:
    """Records the last run() call and replays canned rows."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.last_cypher = None
        self.last_params = None

    def run(self, cypher, **params):
        self.last_cypher = cypher
        self.last_params = params
        return iter([dict(r) for r in self.rows])


def test_registry_non_empty_and_keyed_by_name():
    assert templates.TEMPLATES
    for name, t in templates.TEMPLATES.items():
        assert isinstance(t, QueryTemplate)
        assert t.name == name


def test_headline_author_paper_topic_institution_pattern_present():
    # The milestone's named multi-hop chain.
    t = templates.TEMPLATES["peer_institutions_via_topics"]
    assert "author_name" in t.params
    cy = t.cypher
    assert "STUDIES" in cy and "WORKS_AT" in cy and "AUTHORED_BY" in cy


@pytest.mark.parametrize("t", templates.list_templates(), ids=lambda t: t.name)
def test_template_is_well_formed(t):
    placeholders = t.placeholders()
    # Every declared param appears in the Cypher.
    for p in t.params:
        assert p in placeholders, f"{t.name}: declared param ${p} not used"
    # Every placeholder is either a declared param or the reserved $limit.
    for ph in placeholders:
        assert ph in t.params or ph in templates._RESERVED_PARAMS, (
            f"{t.name}: undeclared placeholder ${ph}"
        )
    # Results are always bounded.
    assert "limit" in placeholders, f"{t.name}: not LIMIT-bounded"
    assert "limit $limit" in t.cypher.lower()


def test_run_template_injects_limit_and_passes_params():
    sess = _FakeSession(rows=[{"topic": "NLP", "papers": 3}])
    rows = run_template(sess, "author_topics", {"author_name": "Ada Lovelace"}, limit=7)
    assert rows == [{"topic": "NLP", "papers": 3}]
    assert sess.last_params == {"author_name": "Ada Lovelace", "limit": 7}
    assert sess.last_cypher == templates.TEMPLATES["author_topics"].cypher


def test_run_template_default_limit():
    sess = _FakeSession()
    run_template(sess, "author_topics", {"author_name": "x"})
    assert sess.last_params["limit"] == templates.DEFAULT_LIMIT


def test_run_template_missing_param_raises():
    sess = _FakeSession()
    with pytest.raises(ValueError, match="missing params"):
        run_template(sess, "author_topics", {})


def test_run_template_unknown_param_raises():
    sess = _FakeSession()
    with pytest.raises(ValueError, match="unknown params"):
        run_template(sess, "author_topics", {"author_name": "x", "bogus": 1})


def test_run_template_unknown_name_raises():
    sess = _FakeSession()
    with pytest.raises(KeyError):
        run_template(sess, "does_not_exist", {})


def test_render_rows_handles_empty_and_rows():
    assert "no results" in render_rows([])
    out = render_rows([{"institution": "Google", "papers": 12}])
    assert "institution" in out and "Google" in out and "12" in out
