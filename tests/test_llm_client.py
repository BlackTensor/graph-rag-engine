"""Tests for the reusable Ollama wrapper + strict template (M8.1).

The prompt template, reasoning-trace stripping, prompt assembly, and the
chat-call wiring are covered with pure logic / monkeypatching — no live Ollama.
M8.3 exercises a real model end-to-end when picking the production pick.
"""

import pytest

from llm import client


def test_strip_think_removes_reasoning_trace():
    raw = "<think>let me reason about this</think>The answer is 42."
    assert client.strip_think(raw) == "The answer is 42."


def test_strip_think_handles_multiline_and_no_trace():
    assert client.strip_think("<think>\na\nb\n</think>  hello ") == "hello"
    assert client.strip_think("plain answer") == "plain answer"


def test_build_prompt_grounds_on_context_and_question():
    prompt = client.build_prompt("What is X?", "alpha facts")
    assert "Context:\nalpha facts" in prompt
    assert "Question: What is X?" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_build_prompt_handles_empty_context():
    assert "(no context retrieved)" in client.build_prompt("Q?", "")
    assert "(no context retrieved)" in client.build_prompt("Q?", "   ")


def test_system_prompt_is_strict():
    # The hardened template must forbid outside knowledge and give one canonical
    # refusal, and must mention both context modalities (graph facts + passages).
    sp = client.SYSTEM_PROMPT
    assert "ONLY" in sp
    assert client.INSUFFICIENT_CONTEXT in sp
    assert "outside" in sp
    assert "knowledge-graph" in sp
    assert "passages" in sp


def test_generate_wires_messages_and_strips_think(monkeypatch):
    captured = {}

    class FakeMessage:
        content = "<think>reasoning</think>Grounded answer."

    class FakeResp:
        message = FakeMessage()

    class FakeClient:
        def __init__(self, host=None):
            captured["host"] = host

        def chat(self, model, messages, options):
            captured["model"] = model
            captured["messages"] = messages
            captured["options"] = options
            return FakeResp()

    fake_ollama = type(
        "M", (), {"Client": FakeClient, "ResponseError": type("RE", (Exception,), {})}
    )
    monkeypatch.setitem(__import__("sys").modules, "ollama", fake_ollama)

    out = client.generate("PROMPT", system="SYS", model="m1", host="http://h:1")
    assert out == "Grounded answer."  # think stripped
    assert captured["model"] == "m1"
    assert captured["host"] == "http://h:1"
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["messages"][1] == {"role": "user", "content": "PROMPT"}
    # deterministic decoding
    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["seed"] == 0


def test_generate_wraps_connection_error_as_llmerror(monkeypatch):
    class FakeClient:
        def __init__(self, host=None):
            pass

        def chat(self, *a, **k):
            raise ConnectionError("refused")

    fake_ollama = type(
        "M", (), {"Client": FakeClient, "ResponseError": type("RE", (Exception,), {})}
    )
    monkeypatch.setitem(__import__("sys").modules, "ollama", fake_ollama)

    with pytest.raises(client.LLMError):
        client.generate("PROMPT")


def test_answer_from_context_builds_prompt_then_generates(monkeypatch):
    captured = {}

    def fake_generate(prompt, *, system, model=None, host=None):
        captured["prompt"] = prompt
        captured["system"] = system
        return "the answer"

    monkeypatch.setattr(client, "generate", fake_generate)

    out = client.answer_from_context("Who cites whom?", "graph facts here")
    assert out == "the answer"
    assert "graph facts here" in captured["prompt"]
    assert "Who cites whom?" in captured["prompt"]
    assert captured["system"] is client.SYSTEM_PROMPT
