#!/usr/bin/env python3
"""Upload the latest agent trace JSON to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def _latest_trace(traces_dir: Path) -> Path:
    files = sorted(traces_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No trace files in {traces_dir}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload agent trace to HF dataset")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path(os.environ.get("AGENT_TRACES_DIR", "outputs/traces")),
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF dataset repo, e.g. username/build-small-agent-traces",
    )
    parser.add_argument("--trace", type=Path, default=None, help="Specific trace file")
    args = parser.parse_args()

    trace_path = args.trace or _latest_trace(args.traces_dir)
    data = json.loads(trace_path.read_text())

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=trace_path.read_bytes(),
        path_in_repo=f"traces/{trace_path.name}",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Add agent trace {trace_path.stem}",
    )

    print(f"Uploaded {trace_path} -> {args.repo_id}/traces/{trace_path.name}")
    print(f"Skill: {data.get('skill')} | Model: {data.get('model')} | Run: {data.get('run_id')}")


if __name__ == "__main__":
    main()
