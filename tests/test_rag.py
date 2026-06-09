"""Tests for the baseline vector-RAG pipeline (M5.3).

Prompt assembly, reasoning-trace stripping, and the retrieve->ground->answer
wiring are covered with pure logic / monkeypatching (no Qdrant or Ollama). The
live end-to-end run is M5.4.
"""

from vector import rag


def test_strip_think_removes_reasoning_trace():
    raw = "<think>let me reason about this</think>The answer is 42."
    assert rag._strip_think(raw) == "The answer is 42."


def test_strip_think_handles_multiline_and_no_trace():
    assert rag._strip_think("<think>\na\nb\n</think>  hello ") == "hello"
    assert rag._strip_think("plain answer") == "plain answer"


def test_format_context_numbers_passages():
    ctx = rag.format_context(
        [
            {"title": "Paper A", "text": "alpha"},
            {"title": "Paper B", "text": "beta"},
        ]
    )
    assert "[1] Paper A\nalpha" in ctx
    assert "[2] Paper B\nbeta" in ctx


def test_build_prompt_grounds_on_context_and_question():
    prompt = rag.build_prompt("What is X?", [{"title": "T", "text": "body"}])
    assert "Context passages:" in prompt
    assert "[1] T\nbody" in prompt
    assert "Question: What is X?" in prompt


def test_build_prompt_handles_empty_context():
    prompt = rag.build_prompt("Q?", [])
    assert "(no context retrieved)" in prompt


def test_system_prompt_constrains_to_context():
    # The baseline must instruct the model not to use outside knowledge.
    assert "ONLY" in rag.SYSTEM_PROMPT
    assert "don't know" in rag.SYSTEM_PROMPT


def test_vector_rag_wires_retrieval_prompt_and_generation(monkeypatch):
    hits = [
        {"title": "Attention Is All You Need", "text": "transformer", "score": 0.9,
         "paper_id": "W1", "chunk_index": 0},
    ]
    captured = {}

    monkeypatch.setattr(rag.search, "search", lambda query, k, client=None: hits)

    def fake_generate(prompt, model=None, host=None):
        captured["prompt"] = prompt
        return "Grounded answer."

    monkeypatch.setattr(rag, "generate", fake_generate)

    result = rag.vector_rag("What is the Transformer?", k=3)
    assert result["query"] == "What is the Transformer?"
    assert result["answer"] == "Grounded answer."
    assert result["contexts"] is hits
    # the retrieved chunk text must have reached the prompt the LLM saw
    assert "transformer" in captured["prompt"]
    assert "What is the Transformer?" in captured["prompt"]
