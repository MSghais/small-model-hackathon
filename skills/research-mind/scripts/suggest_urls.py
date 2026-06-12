#!/usr/bin/env python3
"""CLI stub: URL suggestion requires a loaded inference backend (use Gradio/agent)."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suggest URLs for a topic (use agent runner for full flow)"
    )
    parser.add_argument("topic", help="Research topic")
    args = parser.parse_args()
    print(
        "Use AgentRunner.run_researchmind_discover() or the Gradio Research tab "
        f"to suggest URLs for: {args.topic!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
