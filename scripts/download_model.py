#!/usr/bin/env python3
"""Download a configured model preset from Hugging Face Hub for offline dev."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from inference.config import get_app_config, get_model_config


def _is_local_path(value: str) -> bool:
    return value.startswith(("./", "../")) or Path(value).is_absolute()


def _download_llama_cpp(model, preset_key: str, output_dir: Path) -> Path:
    if model.model_path:
        path = Path(model.model_path)
        if not path.exists():
            raise SystemExit(f"Local MODEL_PATH does not exist: {model.model_path}")
        print(f"Preset {preset_key!r} already points to local file: {path}")
        print(f"Set MODEL_PATH={path} or update models.yaml model_path to use it directly.")
        return path.resolve()

    if not model.model_repo or not model.model_file:
        raise SystemExit(f"Preset {preset_key!r} is missing model_repo/model_file.")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=model.model_repo,
            filename=model.model_file,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
    )
    print(f"Preset: {preset_key} ({model.label})")
    print(f"Model ready at: {path}")
    print("Add to models.yaml under that preset:")
    print(f"  model_path: {path.resolve()}")
    return path.resolve()


def _download_transformers(model, preset_key: str, output_dir: Path) -> Path:
    if not model.model_id:
        raise SystemExit(f"Preset {preset_key!r} is missing model_id.")

    if _is_local_path(model.model_id):
        path = Path(model.model_id).resolve()
        if not path.exists():
            raise SystemExit(f"Local model_id does not exist: {model.model_id}")
        print(f"Preset {preset_key!r} already points to local path: {path}")
        print(f"Set MODEL_ID={path} or update models.yaml model_id to use it directly.")
        return path

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / model.model_id
    path = Path(
        snapshot_download(
            repo_id=model.model_id,
            local_dir=dest,
            local_dir_use_symlinks=False,
        )
    )
    print(f"Preset: {preset_key} ({model.label})")
    print(f"Model ready at: {path}")
    print("Add to models.yaml under that preset:")
    print(f"  model_id: ./{path.relative_to(Path.cwd())}")
    return path.resolve()


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
        help="Directory to download the model into",
    )
    args = parser.parse_args()

    app_config = get_app_config()
    preset_key = args.preset or app_config.active_model
    model = get_model_config(preset_key)

    if model.backend == "llama_cpp":
        _download_llama_cpp(model, preset_key, args.output_dir)
        return

    if model.backend == "transformers":
        _download_transformers(model, preset_key, args.output_dir)
        return

    raise SystemExit(f"Preset {preset_key!r} uses unsupported backend {model.backend!r}.")


if __name__ == "__main__":
    main()
