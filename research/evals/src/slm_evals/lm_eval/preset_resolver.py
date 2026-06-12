"""Resolve models.yaml presets and paths into lm-eval model specifications."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[5]


def _ensure_inference_on_path() -> None:
    libs = _REPO_ROOT / "libs" / "inference" / "src"
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))


def _is_ensemble_checkpoint(path: str | Path) -> bool:
    try:
        from ensemble.checkpoint import is_ensemble_checkpoint

        return is_ensemble_checkpoint(path)
    except ImportError:
        return (Path(path) / "manifest.json").is_file()


def _is_lm_evalable_preset(model) -> bool:
    if model.backend != "transformers":
        return False
    if model.multimodal:
        return False
    if not model.model_id:
        return False
    return True


@dataclass(frozen=True)
class LMEvalModelSpec:
    """Resolved model target for lm-evaluation-harness."""

    lm_eval_model: str
    model_args: dict[str, Any]
    preset_key: str | None
    base_model: str
    adapter_path: str | None
    checkpoint_path: str | None
    trust_remote_code: bool

    def model_args_string(self) -> str:
        parts = []
        for key, value in self.model_args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                parts.append(f"{key}={str(value).lower()}")
            else:
                parts.append(f"{key}={value}")
        return ",".join(parts)


def resolve_model_spec(
    *,
    preset: str | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    trust_remote_code: bool | None = None,
    dtype: str | None = None,
    device: str | None = None,
) -> LMEvalModelSpec:
    """Resolve preset or explicit paths into an lm-eval model specification."""
    if preset and model_path:
        raise ValueError("Pass only one of --preset or --model, not both.")

    if preset:
        return _resolve_from_preset(
            preset,
            adapter_override=adapter_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device=device,
        )

    if not model_path:
        raise ValueError("One of --preset or --model is required.")

    path = Path(model_path)
    if not path.is_absolute() and not str(model_path).startswith(
        ("openbmb/", "google/", "meta-llama/", "Qwen/", "HuggingFaceTB/")
    ):
        path = (_REPO_ROOT / path).resolve()
        model_path = str(path)

    if _is_ensemble_checkpoint(model_path):
        args: dict[str, Any] = {"checkpoint_path": model_path}
        if dtype:
            args["dtype"] = dtype
        if device:
            args["device"] = device
        return LMEvalModelSpec(
            lm_eval_model="ensemble-lm",
            model_args=args,
            preset_key=None,
            base_model=model_path,
            adapter_path=None,
            checkpoint_path=model_path,
            trust_remote_code=trust_remote_code or False,
        )

    base = model_path
    peft = adapter_path
    if peft and not Path(peft).is_absolute():
        peft = str((_REPO_ROOT / peft).resolve())

    args = {
        "pretrained": base,
        "trust_remote_code": trust_remote_code
        if trust_remote_code is not None
        else True,
    }
    if peft:
        args["peft"] = peft
    if dtype:
        args["dtype"] = dtype

    return LMEvalModelSpec(
        lm_eval_model="hf",
        model_args=args,
        preset_key=None,
        base_model=base,
        adapter_path=peft,
        checkpoint_path=None,
        trust_remote_code=bool(args["trust_remote_code"]),
    )


def _resolve_from_preset(
    preset_key: str,
    *,
    adapter_override: str | None,
    trust_remote_code: bool | None,
    dtype: str | None,
    device: str | None,
) -> LMEvalModelSpec:
    _ensure_inference_on_path()
    from inference.config import get_app_config, get_model_config

    app_config = get_app_config(reload=True)
    if preset_key not in app_config.models:
        raise ValueError(
            f"Unknown preset {preset_key!r}. "
            f"Available: {', '.join(sorted(app_config.models))}"
        )

    model = get_model_config(preset_key).resolve_paths(_REPO_ROOT)
    if not _is_lm_evalable_preset(model):
        raise ValueError(
            f"Preset {preset_key!r} uses backend={model.backend!r} "
            f"(multimodal={model.multimodal}); only text transformers presets "
            "are supported for lm-eval."
        )

    model_id = model.model_id
    if not model_id:
        raise ValueError(f"Preset {preset_key!r} has no model_id.")

    if _is_ensemble_checkpoint(model_id):
        args: dict[str, Any] = {"checkpoint_path": model_id}
        if dtype:
            args["dtype"] = dtype
        if device:
            args["device"] = device
        return LMEvalModelSpec(
            lm_eval_model="ensemble-lm",
            model_args=args,
            preset_key=preset_key,
            base_model=model_id,
            adapter_path=None,
            checkpoint_path=model_id,
            trust_remote_code=model.trust_remote_code,
        )

    adapter = adapter_override or model.adapter_path
    trust = (
        trust_remote_code
        if trust_remote_code is not None
        else model.trust_remote_code
    )
    args = {"pretrained": model_id, "trust_remote_code": trust}
    if adapter:
        args["peft"] = adapter
    if dtype:
        args["dtype"] = dtype

    return LMEvalModelSpec(
        lm_eval_model="hf",
        model_args=args,
        preset_key=preset_key,
        base_model=model_id,
        adapter_path=adapter,
        checkpoint_path=None,
        trust_remote_code=trust,
    )
