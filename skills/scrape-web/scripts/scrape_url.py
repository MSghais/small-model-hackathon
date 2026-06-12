#!/usr/bin/env python3
"""CLI: scrape a URL and print extracted title + text preview."""

from __future__ import annotations

import argparse
import sys

from researchmind.scrape_web import fetch_and_extract


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape a URL for ResearchMind")
    parser.add_argument("url", help="HTTPS URL to fetch")
    parser.add_argument("--out", help="Write full text to this file")
    args = parser.parse_args()

    doc = fetch_and_extract(args.url)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(doc.text, encoding="utf-8")
    print(f"Title: {doc.title}")
    print(f"URI: {doc.uri}")
    print(f"Chars: {len(doc.text)}")
    if not args.out:
        print(doc.text[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
