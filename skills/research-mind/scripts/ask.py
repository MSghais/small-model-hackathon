#!/usr/bin/env python3
"""CLI stub: Q&A requires a loaded inference backend (use Gradio/agent)."""

from __future__ import annotations

import argparse
import sys

from researchmind.config import get_config
from researchmind.ingest import IngestPipeline
from researchmind.retrieve import retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview retrieval for a question")
    parser.add_argument("question", help="Question to retrieve context for")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config()
    store = IngestPipeline().store
    chunks = retrieve(args.question, store, config=cfg, top_k=args.top_k)
    if not chunks:
        print("No chunks in store. Ingest sources first.")
        return 1
    for i, c in enumerate(chunks, 1):
        print(f"\n--- [{i}] {c.doc_title} ---\n{c.text[:500]}...")
    print("\nUse AgentRunner.run_researchmind_chat() for a full cited answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
