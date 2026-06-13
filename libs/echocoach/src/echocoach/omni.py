"""Optional MiniCPM-o omni speech-in/speech-out backend (GPU-only, falls back to pipeline)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def voice_profile() -> str:
    return os.environ.get("ECHOCOACH_VOICE_PROFILE", "pipeline").strip().lower()


def is_omni_profile() -> bool:
    return voice_profile() == "omni"


def try_omni_turn(
    audio_path: str,
    *,
    language: str,
    history: list,
    system_prompt: str,
) -> tuple[str | None, str | None, str | None]:
    """Attempt speech-in/speech-out via MiniCPM-o. Returns (user_text, reply_text, wav_path) or Nones."""
    if not is_omni_profile():
        return None, None, None

    model_id = os.environ.get("ECHOCOACH_OMNI_MODEL", "openbmb/MiniCPM-o-4_5")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None, None, "Omni profile requires transformers and torch."

    if not torch.cuda.is_available():
        return None, None, "Omni profile requires CUDA; falling back to ASR+LLM+TTS pipeline."

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            init_audio=True,
            init_tts=True,
        ).eval()
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Omni model load failed: {exc}"

    # MiniCPM-o APIs vary by release; pipeline fallback is the supported path until wired.
    _ = (tokenizer, model, audio_path, language, history, system_prompt)
    return None, None, (
        f"Omni preset ({model_id}) is configured but end-to-end omni turn is not wired yet; "
        "using ASR + text LLM + Piper pipeline."
    )


def omni_status_message() -> str | None:
    if not is_omni_profile():
        return None
    model_id = os.environ.get("ECHOCOACH_OMNI_MODEL", "openbmb/MiniCPM-o-4_5")
    return (
        f"**Voice profile:** `omni` ({model_id}). "
        "Falls back to ASR + LLM + Piper until omni turn API is fully integrated."
    )
