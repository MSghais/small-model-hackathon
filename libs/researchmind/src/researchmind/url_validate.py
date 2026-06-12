from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

# arXiv IDs look like 2301.00001 or 2301.00001v2
_ARXIV_ABS = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?/?$",
    re.IGNORECASE,
)


def normalize_url(url: str) -> str:
    cleaned = url.strip().strip("\"'<>")
    if not cleaned:
        return ""
    if cleaned.startswith("//"):
        cleaned = "https:" + cleaned
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned
    parsed = urlparse(cleaned)
    if not parsed.netloc:
        return ""
    return parsed.geturl().split("#")[0].rstrip("/")


def is_well_formed(url: str) -> tuple[bool, str]:
    if not url:
        return False, "empty url"
    if "..." in url or "…" in url:
        return False, "truncated placeholder"
    if " " in url:
        return False, "contains spaces"

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"unsupported scheme {parsed.scheme!r}"
    host = parsed.netloc.lower()
    if not host or "." not in host:
        return False, "missing host"
    if host in ("localhost", "127.0.0.1"):
        return False, "local url"

    path = parsed.path or ""
    if "arxiv.org" in host and "/abs/" in path:
        if not _ARXIV_ABS.match(url):
            return False, "invalid arxiv abs url"

    if "ieeexplore.ieee.org" in host and path.rstrip("/") in ("", "/document"):
        return False, "incomplete ieee document url"

    return True, "ok"


def check_reachable(url: str, *, timeout: float = 12.0) -> tuple[bool, str]:
    headers = {"User-Agent": "ResearchMind/0.1 (url-validator)"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            response = client.head(url)
            if response.status_code in (405, 501):
                response = client.get(url)
            if response.status_code >= 400:
                return False, f"http {response.status_code}"
        return True, "ok"
    except httpx.HTTPError as exc:
        return False, str(exc)


def validate_url(url: str, *, check_reachable: bool = True) -> tuple[bool, str, str]:
    """Return (ok, reason, normalized_url)."""
    normalized = normalize_url(url)
    ok, reason = is_well_formed(normalized)
    if not ok:
        return False, reason, normalized
    if check_reachable:
        ok, reason = check_reachable(normalized)
        if not ok:
            return False, reason, normalized
    return True, "ok", normalized


def filter_valid_urls(
    urls: list[str],
    *,
    check_reachable: bool = True,
    max_results: int = 5,
) -> list[str]:
    seen: set[str] = set()
    valid: list[str] = []
    for raw in urls:
        ok, _reason, normalized = validate_url(raw, check_reachable=check_reachable)
        if ok and normalized not in seen:
            seen.add(normalized)
            valid.append(normalized)
        if len(valid) >= max_results:
            break
    return valid
