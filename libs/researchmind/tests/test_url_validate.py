from __future__ import annotations

from researchmind.url_validate import (
    filter_valid_urls,
    is_well_formed,
    normalize_url,
    validate_url,
)


def test_rejects_truncated_and_bad_arxiv():
    ok, reason = is_well_formed("https://arxiv.org/abs/quantcomm/2021/10.0")
    assert not ok
    assert "arxiv" in reason

    ok, reason = is_well_formed("https://ieeexplore.ieee.org/document/...")
    assert not ok


def test_accepts_valid_arxiv():
    ok, _ = is_well_formed("https://arxiv.org/abs/2301.00001")
    assert ok


def test_normalize_adds_scheme():
    assert normalize_url("en.wikipedia.org/wiki/AI_agent").startswith("https://")


def test_validate_url_does_not_shadow_probe(monkeypatch):
    """Regression: check_reachable=True must not call the bool parameter."""

    def fake_probe(url, *, timeout=12.0):
        return True, "ok"

    monkeypatch.setattr("researchmind.url_validate.probe_url_reachable", fake_probe)
    ok, reason, normalized = validate_url(
        "https://en.wikipedia.org/wiki/Agent",
        check_reachable=True,
    )
    assert ok
    assert reason == "ok"
    assert "wikipedia" in normalized


def test_filter_valid_urls_skips_bad(monkeypatch):
    def fake_validate(url, *, check_reachable=True):
        if "bad" in url:
            return False, "bad", url
        return True, "ok", url

    monkeypatch.setattr("researchmind.url_validate.validate_url", fake_validate)
    out = filter_valid_urls(
        ["https://good.example/a", "https://bad.example/b"],
        check_reachable=False,
        max_results=5,
    )
    assert out == ["https://good.example/a"]
