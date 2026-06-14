"""Tests for TeacherVoice prompt assembly and message building."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from inference.response_clean import reply_ends_complete_sentence
from echocoach.prompts import PITCH_SYSTEM, resolve_aya_preset, system_prompt_for_mode
from echocoach.teacher_voice import (
    RagContext,
    append_chat_turn,
    build_teacher_messages,
    fetch_rag_context,
    history_to_messages,
)
from echocoach.voiceout import (
    extract_message_text,
    last_assistant_message,
    split_sentences,
    strip_references_for_tts,
)

_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"


class _MockBackend:
    def load(self) -> None:
        pass

    def chat(self, messages, *, max_tokens=512, temperature=0.7):
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        return "Plants use sunlight to make food."

    def generate(self, prompt, *, max_tokens=512, temperature=0.7):
        return self.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


def test_append_chat_turn_messages_format():
    from echocoach.teacher_voice import append_chat_turn

    history = append_chat_turn([], "Hi", "Hello")
    assert history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]

    extended = append_chat_turn(history, "Next?", "Sure.")
    assert extended == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Next?"},
        {"role": "assistant", "content": "Sure."},
    ]


def test_append_chat_turn_migrates_legacy_tuples():
    from echocoach.teacher_voice import append_chat_turn

    legacy = [("Old question", "Old answer")]
    history = append_chat_turn(legacy, "New?", "New reply.")
    assert history[-2:] == [
        {"role": "user", "content": "New?"},
        {"role": "assistant", "content": "New reply."},
    ]
    assert history[0] == {"role": "user", "content": "Old question"}


def test_append_chat_turn_attaches_voice_to_assistant_message(tmp_path):
    wav = tmp_path / "reply.wav"
    wav.write_bytes(b"RIFF")

    history = append_chat_turn(
        [],
        "Hi",
        "Hello",
        assistant_display=f"{_THINK_OPEN}plan{_THINK_CLOSE}\n\nHello",
        voice_path=str(wav),
    )
    assistant = history[-1]
    assert assistant["role"] == "assistant"
    assert isinstance(assistant["content"], list)
    assert assistant["content"][0].startswith(_THINK_OPEN)
    assert assistant["content"][1] == {"path": str(wav)}


def test_history_to_messages_strips_assistant_reasoning():
    history = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": f"{_THINK_OPEN}planning{_THINK_CLOSE}\n\nHello there.",
        },
    ]
    messages = history_to_messages(history)
    assert messages[-1]["content"] == "Hello there."


def test_history_to_messages_tuple_pairs():
    history = [("Hi", "Hello"), ("What is AI?", "Machine learning.")]
    messages = history_to_messages(history)
    assert messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "Machine learning."},
    ]


def test_build_teacher_messages_includes_topic_and_rag():
    rag = RagContext(
        context_block="[1] Plants need light.",
        references_markdown="**References**\n[1] Biology",
        chunk_count=1,
    )
    messages = build_teacher_messages(
        mode="lesson",
        history=[],
        user_text="How do plants eat?",
        topic="Photosynthesis",
        rag=rag,
    )
    assert "TeacherVoice" in messages[0]["content"]
    assert "lesson-planning" in messages[0]["content"]
    assert "Photosynthesis" in messages[0]["content"]
    assert "[1] Plants need light." in messages[-1]["content"]
    assert "How do plants eat?" in messages[-1]["content"]
    assert "Reply now in 2-4 complete spoken sentences only" in messages[-1]["content"]


def test_resolve_aya_preset_uses_global_only():
    assert resolve_aya_preset("fr", "auto") == "tiny-aya-global"
    assert resolve_aya_preset("hi", "auto") == "tiny-aya-global"
    assert resolve_aya_preset("en", "tiny-aya-water") == "tiny-aya-global"


def test_build_teacher_messages_includes_language_instruction():
    messages = build_teacher_messages(
        mode="lesson",
        history=[],
        user_text="Explique le fine-tuning.",
        topic="ML",
        language="fr",
    )
    assert "Target language: French" in messages[0]["content"]
    assert "Reply ONLY in French" in messages[0]["content"]


def test_pitch_mode_system_prompt():
    assert "public-speaking coach" in system_prompt_for_mode("pitch")
    assert PITCH_SYSTEM == system_prompt_for_mode("pitch")


def test_split_sentences():
    text = "Hello there. How are you? Great!"
    assert split_sentences(text) == ["Hello there.", "How are you?", "Great!"]


def test_extract_message_text():
    assert extract_message_text("Hello") == "Hello"
    assert extract_message_text([{"text": "Hello there."}]) == "Hello there."
    assert extract_message_text([{"text": "A"}, {"text": "B"}]) == "A\nB"


def test_last_assistant_message():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there."},
    ]
    assert last_assistant_message(history) == "Hello there."
    assert last_assistant_message([]) is None
    gradio_history = [
        {"role": "user", "content": [{"text": "Hi"}]},
        {"role": "assistant", "content": [{"text": "Hello there."}]},
    ]
    assert last_assistant_message(gradio_history) == "Hello there."


def test_vibevoice_preset_in_voice_models():
    from echocoach.config import get_echo_coach_config

    config = get_echo_coach_config(reload=True)
    preset = config.get_tts("vibevoice-realtime-0.5b")
    assert preset.backend == "vibevoice"
    assert preset.model_id == "microsoft/VibeVoice-Realtime-0.5B"
    assert preset.realtime is True
    assert preset.streaming is True
    assert "en" in preset.supported_languages
    assert config.realtime_tts_preset == "vibevoice-realtime-0.5b"


def test_strip_references_for_tts():
    text = "Answer here.\n\n**References**\n[1] Source"
    assert strip_references_for_tts(text) == "Answer here."


def test_fetch_rag_context_empty_store_warns(research_env):
    ctx = fetch_rag_context("What is photosynthesis?", session_id="", doc_ids=None)
    assert ctx is not None
    assert ctx.chunk_count == 0
    assert ctx.warning


def test_retrieval_query_exported():
    from researchmind.scope import retrieval_query as rm_query

    assert rm_query("step 2?", topic="Photosynthesis") == "Photosynthesis: step 2?"


def test_rag_turn_via_agent_mock(monkeypatch, tmp_path):
    from agent.models import Citation, ResearchChatResult
    from echocoach.teacher_voice import _rag_turn_via_agent
    from agent.trace import TraceRecorder

    result = ResearchChatResult(
        answer="Plants use light [1].\n\n**References**\n[1] Bio",
        citations=[
            Citation(
                index=1,
                chunk_id="c1",
                doc_title="Bio",
                doc_uri="https://example.com",
                excerpt="Plants use light.",
            )
        ],
        references_markdown="**References**\n[1] Bio",
        session_id="",
        trace_path=str(tmp_path / "trace.json"),
    )

    class _RunnerStub:
        def run_researchmind_chat(self, **kwargs):
            return result

    monkeypatch.setattr("echocoach.teacher_voice.AgentRunner", _RunnerStub)

    trace = TraceRecorder(skill="teacher-voice", model="test", user_input={})
    text, refs, status, display = _rag_turn_via_agent(
        "How do plants eat?",
        mode="explain",
        topic="Photosynthesis",
        session_id="",
        doc_ids=None,
        model_key="test",
        backend=_MockBackend(),
        trace=trace,
    )
    assert "Plants use light" in text
    assert refs
    assert "1" in status
    assert display


@pytest.fixture
def research_env(tmp_path, monkeypatch):
    from researchmind.config import ResearchMindConfig

    cfg = ResearchMindConfig(
        data_dir=tmp_path / "rm",
        embed_model="test",
        auto_search=False,
        top_k=2,
        max_context_chunks=8,
        chunk_size=50,
        chunk_overlap=10,
    )
    monkeypatch.setenv("RESEARCHMIND_DATA_DIR", str(cfg.data_dir))
    monkeypatch.setenv("AGENT_OUTPUTS_DIR", str(tmp_path / "outputs"))


def test_finalize_voice_reply_compacts_incomplete_sentence():
    from echocoach.teacher_voice import _finalize_voice_reply
    from agent.trace import TraceRecorder

    class _Backend:
        def chat(self, messages, *, max_tokens=512, temperature=0.2):
            return (
                "Finetuning adapts a pretrained small model to your task using extra labeled data. "
                "You keep most of the base weights and train on a focused dataset. "
                "That usually beats prompting alone for domain-specific work."
            )

    trace = TraceRecorder(skill="teacher-voice", model="test", user_input={})
    text, display = _finalize_voice_reply(
        "The lesson aims to teach how to fine-tune small",
        mode="lesson",
        backend=_Backend(),
        trace=trace,
    )
    assert reply_ends_complete_sentence(text)
    assert "fine-tune" in text.lower() or "finetun" in text.lower()
    assert text == display


def test_run_teacher_voice_text_turn_mock(monkeypatch, tmp_path):
    from echocoach.teacher_voice import run_teacher_voice_text_turn

    class _Tts:
        def synthesize(self, text, *, language, out_dir=None):
            out = (out_dir or tmp_path) / "out.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.zeros(8000, dtype=np.float32), 16_000)
            return str(out), None

    monkeypatch.setattr("echocoach.voiceout.get_tts_backend", lambda _: _Tts())

    result = run_teacher_voice_text_turn(
        "Tell me about plants.",
        [],
        mode="explain",
        backend=_MockBackend(),
        use_rag=False,
    )
    assert result.user_text == "Tell me about plants."
    assert "sunlight" in result.assistant_text
    assert len(result.history) == 2
    assistant = result.history[-1]
    assert assistant["role"] == "assistant"
    assert isinstance(assistant["content"], list)
    assert assistant["content"][0] == "Plants use sunlight to make food."
    assert assistant["content"][1]["path"]
    assert result.trace.get("skill") == "teacher-voice"


def test_run_teacher_voice_turn_mock_asr(monkeypatch, tmp_path):
    from echocoach.teacher_voice import run_teacher_voice_turn

    wav = tmp_path / "turn.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)

    class _Asr:
        def transcribe(self, path, *, language="en"):
            return "Tell me about plants."

    class _Tts:
        def synthesize(self, text, *, language, out_dir=None):
            out = (out_dir or tmp_path) / "out.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.zeros(8000, dtype=np.float32), 16_000)
            return str(out), None

    monkeypatch.setattr("echocoach.teacher_voice.get_asr_backend", lambda _: _Asr())
    monkeypatch.setattr("echocoach.voiceout.get_tts_backend", lambda _: _Tts())

    result = run_teacher_voice_turn(
        str(wav),
        [],
        mode="explain",
        backend=_MockBackend(),
        use_rag=False,
    )
    assert result.user_text == "Tell me about plants."
    assert "sunlight" in result.assistant_text
    assert len(result.history) == 2
    assert result.history[0]["role"] == "user"
    assert result.history[1]["role"] == "assistant"
    assert result.trace.get("skill") == "teacher-voice"
