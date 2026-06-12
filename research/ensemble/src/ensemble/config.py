"""Resolve base LLM for ensemble from .env and models.yaml (same order as finetune)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FALLBACK_PRESET = "minicpm5-1b"

_ENV_LLM_KEYS = (
    "ENSEMBLE_LLM",
    "LLM_PATH",
    "BASE",
    "FINETUNE_MODEL",
    "MODEL_ID",
)


def repo_root() -> Path:
    return _REPO_ROOT


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from repo .env without overriding existing env vars."""
    path = _REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _ensure_inference_on_path() -> None:
    libs = _REPO_ROOT / "libs" / "inference" / "src"
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))


def _is_ensemble_llm_preset(model) -> bool:
    return model.backend == "transformers" and not model.multimodal and bool(
        model.model_id
    )


def _llm_from_local_path(raw: str) -> str | None:
    path = Path(raw)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    if path.suffix == ".gguf":
        return None
    if path.is_dir() and (path / "config.json").is_file():
        return str(path)
    if path.is_file():
        return None
    return None


def _llm_from_env_paths() -> str | None:
    for key in ("LLM_PATH", "MODEL_PATH"):
        raw = os.environ.get(key)
        if raw:
            resolved = _llm_from_local_path(raw)
            if resolved:
                return resolved
    return None


def resolve_llm(
    *,
    llm_arg: str | None = None,
    preset_arg: str | None = None,
) -> tuple[str, str | None]:
    """
    Return (hub_id_or_local_path, preset_key) for ensemble HF backends.

    Priority when llm_arg is None or ``auto``:
      1. ENSEMBLE_LLM, LLM_PATH (local HF dir), BASE, FINETUNE_MODEL, MODEL_ID
      2. MODEL_PATH if it points at a HuggingFace model directory (not .gguf)
      3. ENSEMBLE_PRESET, FINETUNE_PRESET, or ACTIVE_MODEL from models.yaml
      4. First fine-tunable transformers preset (default minicpm5-1b)
    """
    if llm_arg and llm_arg not in ("auto",):
        return llm_arg, preset_arg

    for env_name in _ENV_LLM_KEYS:
        raw = os.environ.get(env_name)
        if raw:
            local = _llm_from_local_path(raw)
            return local or raw, preset_arg

    local = _llm_from_env_paths()
    if local:
        return local, preset_arg

    _ensure_inference_on_path()
    from inference.config import get_app_config, get_model_config

    app_config = get_app_config(reload=True)
    preset_key = (
        preset_arg
        or os.environ.get("ENSEMBLE_PRESET")
        or os.environ.get("FINETUNE_PRESET")
        or os.environ.get("ACTIVE_MODEL")
    )

    if preset_key and preset_key in app_config.models:
        model = get_model_config(preset_key)
        if not _is_ensemble_llm_preset(model):
            preset_key = None

    if preset_key is None:
        for candidate in (_FALLBACK_PRESET, *app_config.models):
            if candidate not in app_config.models:
                continue
            model = get_model_config(candidate)
            if _is_ensemble_llm_preset(model):
                preset_key = candidate
                break

    if not preset_key:
        raise SystemExit(
            "No transformers LLM found for ensemble. Pass --llm, set LLM_PATH/BASE/"
            "MODEL_ID in .env, or ACTIVE_MODEL in models.yaml."
        )

    model = get_model_config(preset_key)
    if not _is_ensemble_llm_preset(model):
        raise SystemExit(
            f"Preset {preset_key!r} cannot back an ensemble "
            f"(backend={model.backend}, multimodal={model.multimodal})."
        )
    return model.model_id, preset_key


def default_ensemble_out(preset_key: str | None) -> str:
    label = preset_key or "custom"
    return str((_REPO_ROOT / "models" / "ensemble" / f"{label}-jepa-pretrain").resolve())


def resolve_llm_cli(
    llm: str | None,
    *,
    toy: bool = False,
    preset: str | None = None,
) -> str:
    """CLI helper: explicit tiny, else .env / models.yaml unless --toy without --llm."""
    if llm == "tiny":
        return "tiny"
    if llm is None or llm == "auto":
        if toy:
            return "tiny"
        load_dotenv()
        resolved, _ = resolve_llm(preset_arg=preset)
        return resolved
    return llm
