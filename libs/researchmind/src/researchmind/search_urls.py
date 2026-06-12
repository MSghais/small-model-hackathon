from __future__ import annotations

import logging

from researchmind.url_validate import filter_valid_urls, normalize_url

logger = logging.getLogger(__name__)


def build_search_queries(topic: str) -> list[str]:
    """Craft Google-friendly queries for a research topic."""
    t = topic.strip()
    if not t:
        return []
    return [
        f"{t} site:wikipedia.org",
        f'"{t}" introduction overview',
        f"{t} tutorial guide site:.edu OR site:.gov",
        f"{t} research paper site:arxiv.org",
        f"what is {t}",
    ]


def _google_search(query: str, *, n: int) -> list[str]:
    urls: list[str] = []
    try:
        from googlesearch import search

        for item in search(query, num_results=n, lang="en", timeout=15):
            if isinstance(item, str):
                urls.append(item)
            else:
                href = getattr(item, "url", None) or getattr(item, "link", None)
                if href:
                    urls.append(str(href))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Google search failed for %r: %s", query, exc)
    return urls


def _duckduckgo_search(query: str, *, n: int) -> list[str]:
    urls: list[str] = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        ddgs = DDGS()
        results = ddgs.text(query, max_results=n)
        if results is None:
            return urls
        for item in results:
            if not isinstance(item, dict):
                continue
            href = item.get("href") or item.get("link")
            if href:
                urls.append(str(href))
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
    return urls


def _collect_candidates(topic: str, *, per_query: int = 4) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for query in build_search_queries(topic):
        batch = _google_search(query, n=per_query)
        if not batch:
            batch = _duckduckgo_search(query, n=per_query)
        for raw in batch:
            normalized = normalize_url(raw)
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)
    return candidates


def search_urls(
    topic: str,
    *,
    n: int = 5,
    check_reachable: bool = True,
) -> list[str]:
    """
    Search the web (Google first, DuckDuckGo fallback) and return verified URLs.
    """
    candidates = _collect_candidates(topic, per_query=max(n, 4))
    return filter_valid_urls(candidates, check_reachable=check_reachable, max_results=n)
