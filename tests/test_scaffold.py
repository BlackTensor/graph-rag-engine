"""Smoke test: the package imports and config exposes sane defaults."""

import config


def test_config_defaults():
    assert config.NEO4J_URI.startswith("bolt://")
    assert config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert config.QDRANT_COLLECTION
