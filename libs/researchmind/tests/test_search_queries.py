from __future__ import annotations

from researchmind.search_urls import build_search_queries, search_urls


def test_build_search_queries_includes_wikipedia_and_arxiv():
    queries = build_search_queries("AI agent")
    joined = " ".join(queries).lower()
    assert "wikipedia" in joined
    assert "arxiv" in joined
    assert "ai agent" in joined


def test_search_urls_uses_validated_results(monkeypatch):
    monkeypatch.setattr(
        "researchmind.search_urls._collect_candidates",
        lambda topic, per_query=4: [
            "https://en.wikipedia.org/wiki/Intelligent_agent",
            "https://arxiv.org/abs/quantcomm/2021/10.0",
        ],
    )

    def fake_filter(urls, *, check_reachable=True, max_results=5):
        return [u for u in urls if "wikipedia" in u][:max_results]

    monkeypatch.setattr("researchmind.search_urls.filter_valid_urls", fake_filter)
    out = search_urls("AI agent", n=3, check_reachable=False)
    assert len(out) == 1
    assert "wikipedia" in out[0]
