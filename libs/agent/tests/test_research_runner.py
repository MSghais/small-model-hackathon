from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent.runner import AgentRunner
from researchmind.config import ResearchMindConfig
from researchmind.extract import ExtractedDocument
from researchmind.store import MemRAGStore


class MockBackend:
    def load(self) -> None:
        return None

    def chat(self, messages, *, max_tokens=512, temperature=0.7):
        user = messages[-1]["content"]
        if "Topic:" in user:
            return '["https://example.com/a", "https://example.com/b"]'
        return "Plants use photosynthesis [1]."

    def generate(self, prompt, *, max_tokens=512, temperature=0.7):
        return self.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


@pytest.fixture
def research_env(tmp_path, monkeypatch):
    cfg = ResearchMindConfig(
        data_dir=tmp_path / "rm",
        embed_model="test",
        auto_search=False,
        top_k=2,
        chunk_size=50,
        chunk_overlap=10,
    )
    monkeypatch.setenv("RESEARCHMIND_DATA_DIR", str(cfg.data_dir))

    def fake_embed(texts, *, model_name):
        vecs = []
        for t in texts:
            vecs.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
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


def test_discover_urls(research_env):
    runner = AgentRunner()
    result = runner.run_researchmind_discover(
        topic="photosynthesis",
        auto_search=False,
        session_id=None,
        model_key="test",
        backend=MockBackend(),
    )
    assert len(result.suggested_urls) >= 1
    assert result.session_id


def test_ingest_and_chat(research_env):
    runner = AgentRunner()
    ingest = runner.run_researchmind_ingest(
        topic=None,
        urls=["https://example.com/a"],
        files=[],
        auto_search=False,
        session_id=None,
        model_key="test",
        backend=MockBackend(),
    )
    assert ingest.doc_count >= 1
    assert ingest.chunk_count >= 1

    chat = runner.run_researchmind_chat(
        question="How do plants make energy?",
        session_id=ingest.session_id,
        model_key="test",
        backend=MockBackend(),
    )
    assert "photosynthesis" in chat.answer.lower() or "[1]" in chat.answer
    assert chat.session_id == ingest.session_id
