#!/usr/bin/env python3
"""CLI: extract text from a PDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from researchmind.scrape_pdf import extract_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract PDF text for ResearchMind")
    parser.add_argument("path", type=Path, help="Path to PDF file")
    parser.add_argument("--out", help="Write full text to this file")
    args = parser.parse_args()

    doc = extract_pdf(args.path)
    if args.out:
        Path(args.out).write_text(doc.text, encoding="utf-8")
    print(f"Title: {doc.title}")
    print(f"Pages metadata: {doc.metadata.get('page_count', '?')}")
    print(f"Chars: {len(doc.text)}")
    if not args.out:
        print(doc.text[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
