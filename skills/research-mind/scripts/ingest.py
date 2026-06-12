#!/usr/bin/env python3
"""CLI: ingest URLs from a file (one per line)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from researchmind.ingest import IngestPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest URLs for ResearchMind")
    parser.add_argument("urls_file", type=Path, help="Text file with one URL per line")
    parser.add_argument("--session", help="Optional session id")
    args = parser.parse_args()

    pipeline = IngestPipeline()
    lines = [ln.strip() for ln in args.urls_file.read_text().splitlines() if ln.strip()]
    for url in lines:
        doc_id, is_new = pipeline.ingest_url(url, session_id=args.session)
        print(f"{url} -> {doc_id} ({'new' if is_new else 'dup'})")
    print(f"Total chunks: {pipeline.store.count_chunks()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
