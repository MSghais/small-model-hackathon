from __future__ import annotations

import json
import numpy as np
import pytest

from agent.models import EducationPptxInput, ResearchIngestResult
from agent.prompts import education_outline_user
from agent.runner import AgentRunner
from researchmind.config import ResearchMindConfig
from researchmind.extract import ExtractedDocument


def _outline_json(slide_count: int = 3) -> str:
    slides = [
        {
            "title": f"Slide {i}",
            "bullets": ["Key point"],
            "speaker_note": "Note",
        }
        for i in range(1, slide_count + 1)
    ]
    return json.dumps({"title": "Test Lesson", "slides": slides})


class OutlineBackend:
    def load(self) -> None:
        return None

    def chat(self, messages, *, max_tokens=2048, temperature=0.3):
        return _outline_json(3)

    def generate(self, prompt, *, max_tokens=512, temperature=0.7):
        return self.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


@pytest.fixture
def research_env(tmp_path, monkeypatch):
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

    def fake_embed(texts, *, model_name):
        vecs = [np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in texts]
        return np.stack(vecs) if vecs else np.zeros((0, 3), dtype=np.float32)

    monkeypatch.setattr("researchmind.ingest.embed_texts", fake_embed)
    monkeypatch.setattr("researchmind.retrieve.embed_texts", fake_embed)

    def fake_scrape(url: str):
        return ExtractedDocument(
            source_type="web",
            uri=url,
            title="Example",
            text="Photosynthesis converts light to energy in plants.",
        )

    monkeypatch.setattr("agent.tools.research_tools.fetch_and_extract", fake_scrape)

    def fake_search(topic, *, n=5, check_reachable=True):
        return [f"https://example.com/{topic.replace(' ', '-')}"]

    monkeypatch.setattr("agent.tools.research_tools.search_urls", fake_search)

    def fake_validate(url, *, check_reachable=True):
        normalized = url if url.startswith("http") else f"https://{url}"
        return True, "ok", normalized

    monkeypatch.setattr("researchmind.url_validate.validate_url", fake_validate)
    return cfg


def test_education_outline_user_includes_source_context():
    req = EducationPptxInput(topic="Photosynthesis", grade="6", slide_count=3)
    user = education_outline_user(req, source_context="[1] Plants use chlorophyll.")
    assert "retrieved source excerpts" in user
    assert "chlorophyll" in user


def test_none_mode_skips_source_summary(research_env):
    runner = AgentRunner()
    result = runner.run_education_pptx(
        topic="Photosynthesis",
        grade="6",
        slide_count=3,
        model_key="test",
        backend=OutlineBackend(),
        source_mode="none",
    )
    assert result.outline.title == "Test Lesson"
    assert result.source_summary == ""


def test_web_auto_calls_ingest_with_auto_search(research_env, monkeypatch):
    calls: list[dict] = []

    def fake_ingest(self, **kwargs):
        calls.append(kwargs)
        return ResearchIngestResult(
            session_id="sess-auto",
            ingested=["https://example.com/photosynthesis"],
            skipped=[],
            failures=[],
            doc_count=1,
            chunk_count=1,
            trace_path="/tmp/trace.json",
            message="Ingested 1 source(s)",
        )

    monkeypatch.setattr(AgentRunner, "run_researchmind_ingest", fake_ingest)

    runner = AgentRunner()
    result = runner.run_education_pptx(
        topic="Photosynthesis",
        grade="6",
        slide_count=3,
        model_key="test",
        backend=OutlineBackend(),
        source_mode="web",
        search_workflow="auto",
    )
    assert len(calls) == 1
    assert calls[0]["auto_search"] is True
    assert "Ingested 1 source(s)" in result.source_summary


def test_web_two_step_requires_urls(research_env):
    runner = AgentRunner()
    with pytest.raises(ValueError, match="Two-step web search requires"):
        runner.run_education_pptx(
            topic="Photosynthesis",
            grade="6",
            slide_count=3,
            model_key="test",
            backend=OutlineBackend(),
            source_mode="web",
            search_workflow="two_step",
            urls=[],
            files=[],
        )


def test_web_two_step_ingests_without_auto_search(research_env, monkeypatch):
    calls: list[dict] = []

    def fake_ingest(self, **kwargs):
        calls.append(kwargs)
        return ResearchIngestResult(
            session_id="sess-two",
            ingested=["https://example.com/a"],
            skipped=[],
            failures=[],
            doc_count=1,
            chunk_count=1,
            trace_path="/tmp/trace.json",
            message="Ingested 1 source(s)",
        )

    monkeypatch.setattr(AgentRunner, "run_researchmind_ingest", fake_ingest)

    runner = AgentRunner()
    runner.run_education_pptx(
        topic="Photosynthesis",
        grade="6",
        slide_count=3,
        model_key="test",
        backend=OutlineBackend(),
        source_mode="web",
        search_workflow="two_step",
        urls=["https://example.com/a"],
    )
    assert calls[0]["auto_search"] is False


def test_rag_requires_indexed_sources(research_env):
    runner = AgentRunner()
    with pytest.raises(ValueError, match="RAG mode requires indexed sources"):
        runner.run_education_pptx(
            topic="Photosynthesis",
            grade="6",
            slide_count=3,
            model_key="test",
            backend=OutlineBackend(),
            source_mode="rag",
            session_id="",
            urls=[],
            files=[],
        )


def test_rag_uses_session_without_auto_search(research_env, monkeypatch):
    ingest = AgentRunner().run_researchmind_ingest(
        topic="Photosynthesis",
        urls=["https://example.com/a"],
        files=[],
        auto_search=False,
        session_id=None,
        model_key="test",
        backend=OutlineBackend(),
    )

    calls: list[dict] = []

    def fake_ingest(self, **kwargs):
        calls.append(kwargs)
        return ingest

    monkeypatch.setattr(AgentRunner, "run_researchmind_ingest", fake_ingest)

    runner = AgentRunner()
    result = runner.run_education_pptx(
        topic="Photosynthesis",
        grade="6",
        slide_count=3,
        model_key="test",
        backend=OutlineBackend(),
        source_mode="rag",
        session_id=ingest.session_id,
    )
    assert calls == []
    assert "Retrieved" in result.source_summary
