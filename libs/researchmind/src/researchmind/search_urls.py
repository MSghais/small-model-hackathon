from __future__ import annotations


def search_urls(query: str, *, n: int = 5) -> list[str]:
    from duckduckgo_search import DDGS

    urls: list[str] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=n):
            href = item.get("href") or item.get("link")
            if href and href.startswith("http"):
                urls.append(href)
    return urls[:n]
