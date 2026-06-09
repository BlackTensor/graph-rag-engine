"""Integration smoke tests for the local stack.

These require `docker compose up -d`. If a service is unreachable the test is
SKIPPED (not failed), so `pytest` stays green in environments without Docker.
"""

import pytest

import health


@pytest.mark.parametrize(
    "check",
    [health.ping_neo4j, health.ping_qdrant, health.ping_ollama],
    ids=["neo4j", "qdrant", "ollama"],
)
def test_service_reachable(check):
    ok, msg = check()
    if not ok:
        pytest.skip(msg)
    assert ok, msg
