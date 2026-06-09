"""Connectivity smoke checks for Neo4j, Qdrant, and Ollama.

Run after `docker compose up -d`:

    python src/health.py

Exits 0 if all three services respond, 1 otherwise. Client libraries are
imported lazily so this module is cheap to import in tests.
"""

from __future__ import annotations

import sys

import config


def ping_neo4j() -> tuple[bool, str]:
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        try:
            with driver.session() as session:
                ok = session.run("RETURN 1 AS ok").single()["ok"] == 1
        finally:
            driver.close()
        return ok, f"Neo4j OK ({config.NEO4J_URI})"
    except Exception as exc:  # noqa: BLE001 - smoke check reports any failure
        return False, f"Neo4j FAIL ({config.NEO4J_URI}): {exc}"


def ping_qdrant() -> tuple[bool, str]:
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        n = len(client.get_collections().collections)
        return True, f"Qdrant OK ({config.QDRANT_URL}); collections={n}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Qdrant FAIL ({config.QDRANT_URL}): {exc}"


def ping_ollama() -> tuple[bool, str]:
    import ollama

    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        resp = client.list()
        models = [getattr(m, "model", None) or m.get("model") for m in resp.models]
        return True, f"Ollama OK ({config.OLLAMA_HOST}); models={models}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama FAIL ({config.OLLAMA_HOST}): {exc}"


def main() -> int:
    all_ok = True
    for passed, msg in (ping_neo4j(), ping_qdrant(), ping_ollama()):
        print(("[ok] " if passed else "[!!] ") + msg)
        all_ok = all_ok and passed
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
