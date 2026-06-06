#!/usr/bin/env python3
"""Download the configured GGUF model from Hugging Face Hub for offline dev."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("MODEL_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF"),
        help="Hugging Face repo containing the GGUF file",
    )
    parser.add_argument(
        "--file",
        default=os.environ.get("MODEL_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf"),
        help="GGUF filename inside the repo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to copy/symlink the downloaded model into",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=args.repo,
        filename=args.file,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model ready at: {path}")
    print(f"Set MODEL_PATH={path} to use this file directly.")


if __name__ == "__main__":
    main()
