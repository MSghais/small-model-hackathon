from __future__ import annotations

import httpx
import trafilatura

from researchmind.extract import ExtractedDocument


def fetch_and_extract(url: str, *, timeout: float = 30.0) -> ExtractedDocument:
    headers = {
        "User-Agent": "ResearchMind/0.1 (local research agent; hackathon)",
    }
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        html = response.text

    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        output_format="txt",
    )
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = (metadata.title if metadata and metadata.title else url) or url
    text = (extracted or "").strip()
    if not text:
        text = html[:50_000]

    return ExtractedDocument(
        source_type="web",
        uri=url,
        title=title,
        text=text,
        mime="text/html",
        metadata={"final_url": str(response.url)},
    )
