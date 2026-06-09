"""Unit tests for audit helpers (no data file needed)."""

from ingest import audit


def test_short_id_strips_prefix():
    assert audit.short_id("https://openalex.org/W123") == "W123"
    assert audit.short_id(None) is None


def test_pct_formats():
    assert "50.0%" in audit.pct(1, 2)
    assert audit.pct(0, 0) == "0"
