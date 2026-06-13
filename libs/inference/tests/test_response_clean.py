from __future__ import annotations

from inference.response_clean import prepare_display_reply, strip_reasoning_output

_RT_OPEN = "<" + "redacted_thinking" + ">"
_RT_CLOSE = "</" + "redacted_thinking" + ">"
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"


def test_strips_redacted_thinking_block():
    raw = f"{_RT_OPEN}\nplanning...\n{_RT_CLOSE}\n\nThe capital of France is Paris."
    assert strip_reasoning_output(raw) == "The capital of France is Paris."


def test_strips_think_block():
    raw = f"{_THINK_OPEN}\nplanning...\n{_THINK_CLOSE}\n\nAgents use memory [1]."
    assert strip_reasoning_output(raw) == "Agents use memory [1]."


def test_strips_malformed_think_prefix_and_extracts_summary():
    raw = """think> We need to summarize the document. First, identify sources.

Let's draft:

Summary: This review covers AI agent applications, evaluation, and future work [1]."""
    out = strip_reasoning_output(raw)
    assert out.startswith("This review covers")
    assert "We need to summarize" not in out


def test_preserves_normal_answer():
    text = "AI agents combine perception, planning, and action [1]."
    assert strip_reasoning_output(text) == text


def test_prepare_display_reply_wraps_malformed_think_prefix():
    raw = "think> We need to plan the answer.\n\nThe answer is 42."
    out = prepare_display_reply(raw)
    assert out.startswith(_THINK_OPEN)
    assert _THINK_CLOSE in out
    assert "We need to plan the answer." in out
