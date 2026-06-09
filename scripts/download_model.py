#!/usr/bin/env python3
"""Download a configured GGUF preset from Hugging Face Hub for offline dev."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from inference.config import get_app_config, get_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset key from models.yaml (default: ACTIVE_MODEL or app default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to copy/symlink the downloaded model into",
    )
    args = parser.parse_args()

    app_config = get_app_config()
    preset_key = args.preset or app_config.active_model
    model = get_model_config(preset_key)

    if model.backend != "llama_cpp":
        raise SystemExit(
            f"Preset {preset_key!r} uses backend {model.backend!r}. "
            "Only llama_cpp presets with model_repo/model_file can be downloaded."
        )

    if model.model_path:
        path = Path(model.model_path)
        if not path.exists():
            raise SystemExit(f"Local MODEL_PATH does not exist: {model.model_path}")
        print(f"Preset {preset_key!r} already points to local file: {path}")
        print(f"Set MODEL_PATH={path} or update models.yaml model_path to use it directly.")
        return

    if not model.model_repo or not model.model_file:
        raise SystemExit(f"Preset {preset_key!r} is missing model_repo/model_file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=model.model_repo,
        filename=model.model_file,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Preset: {preset_key} ({model.label})")
    print(f"Model ready at: {path}")
    print("Add to models.yaml under that preset:")
    print(f"  model_path: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
