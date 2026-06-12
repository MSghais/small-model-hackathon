#!/usr/bin/env python3
"""CLI: ingest a text file or URL into MemRAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from researchmind.extract import ExtractedDocument
from researchmind.ingest import IngestPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk and index content")
    parser.add_argument("--url", help="Scrape and index URL")
    parser.add_argument("--file", type=Path, help="Index local file")
    parser.add_argument("--session", help="Session id to tag document")
    args = parser.parse_args()

    pipeline = IngestPipeline()
    if args.url:
        doc_id, is_new = pipeline.ingest_url(args.url, session_id=args.session)
    elif args.file:
        doc_id, is_new = pipeline.ingest_path(args.file, session_id=args.session)
    else:
        parser.error("Provide --url or --file")

    status = "indexed" if is_new else "deduplicated"
    print(f"Document {doc_id} ({status}), chunks in store: {pipeline.store.count_chunks()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
