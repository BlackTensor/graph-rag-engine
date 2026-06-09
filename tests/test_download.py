"""Unit tests for the OpenAlex downloader (no network)."""

from ingest import download


def test_build_url_has_required_params():
    url = download.build_url("*")
    assert url.startswith("https://api.openalex.org/works?")
    for key in ("filter=", "select=", "cursor=", "mailto=", "per-page="):
        assert key in url


def test_corpus_filter_defined():
    # The reproducible corpus definition from M2.1 must stay present.
    assert "concepts.id:C204321447" in download.FILTER
    assert "referenced_works" in download.SELECT
