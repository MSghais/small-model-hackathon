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


def test_extracts_final_answer_from_plain_chain_of_thought():
    raw = """First, I need to explain finetuning in plain language. I should keep it concise.

Let me draft:
1. Finetuning adjusts a model for a task.
2. Best practices include good data.

Final answer:

Finetuning small model adjusts a model to improve its performance on a specific task.
For example, fine-tuning a language model can enhance its ability to understand complex queries.
Best practices include using diverse and high-quality data.

That's about 3 sentences. I think it covers it.

Let me write:

Finetuning small model involves training the model with additional data to specialize in a task.
For instance, fine-tuning a computer vision model can improve its object"""
    out = strip_reasoning_output(raw)
    assert out.startswith("Finetuning small model adjusts")
    assert "First, I need" not in out
    assert "Let me draft" not in out
    assert "That's about 3 sentences" not in out


def test_prepare_display_reply_collapses_plain_chain_of_thought():
    raw = """First, I need to plan the answer.

Final answer:

Finetuning teaches a small model to specialize on your task using extra training data."""
    out = prepare_display_reply(raw)
    assert out.startswith(_THINK_OPEN)
    assert _THINK_CLOSE in out
    assert "Finetuning teaches a small model" in out
    assert "First, I need to plan" in out


def test_prepare_display_reply_wraps_malformed_think_prefix():
    raw = "think> We need to plan the answer.\n\nThe answer is 42."
    out = prepare_display_reply(raw)
    assert out.startswith(_THINK_OPEN)
    assert _THINK_CLOSE in out
    assert "We need to plan the answer." in out
