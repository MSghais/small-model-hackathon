"""VibeVoice Realtime TTS backend (streaming, low latency).

Model: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
Falls back to Piper when the model is unavailable or not yet wired end-to-end.
"""

from __future__ import annotations

import uuid
import wave
from pathlib import Path
from typing import Protocol

from echocoach.config import TtsPreset, get_echo_coach_config, outputs_dir

_VIBEVOICE_DOC = "https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B"


class TtsBackend(Protocol):
    def synthesize(self, text: str, *, language: str, out_dir: Path | None = None) -> tuple[str | None, str | None]: ...


_vibevoice_cache: dict[str, "VibeVoiceTtsBackend"] = {}


class VibeVoiceTtsBackend:
    def __init__(self, preset: TtsPreset) -> None:
        self._preset = preset
        self._pipeline = None
        self._load_error: str | None = None

    @property
    def model_id(self) -> str:
        return self._preset.model_id or "microsoft/VibeVoice-Realtime-0.5B"

    def _resolve_language(self, language: str) -> tuple[str, str | None]:
        supported = set(self._preset.supported_languages)
        if supported and language not in supported:
            fallback = self._preset.fallback_voice
            return fallback, (
                f"Language {language!r} is not listed for VibeVoice; using {fallback!r}."
            )
        return language, None

    def _try_load(self) -> str | None:
        if self._pipeline is not None:
            return None
        if self._load_error is not None:
            return self._load_error

        try:
            from transformers import pipeline
        except ImportError:
            self._load_error = "VibeVoice requires transformers and torch."
            return self._load_error

        try:
            self._pipeline = pipeline(
                "text-to-speech",
                model=self.model_id,
                trust_remote_code=True,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            self._load_error = (
                f"VibeVoice load failed ({self.model_id}): {exc}. "
                f"See {_VIBEVOICE_DOC} for setup; using Piper fallback."
            )
            return self._load_error

    def _write_wav(self, audio: object, sample_rate: int, out_path: Path) -> None:
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.squeeze()
        pcm = np.clip(samples, -1.0, 1.0)
        pcm_i16 = (pcm * 32767).astype(np.int16)
        with wave.open(str(out_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_i16.tobytes())

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        out_dir: Path | None = None,
    ) -> tuple[str | None, str | None]:
        if not text.strip():
            return None, "No text to synthesize."

        lang, lang_warning = self._resolve_language(language)
        load_warning = self._try_load()
        if load_warning or self._pipeline is None:
            return self._piper_fallback(text, language=lang, out_dir=out_dir, prefix=load_warning)

        base = out_dir or outputs_dir()
        base.mkdir(parents=True, exist_ok=True)
        out_path = base / f"vibevoice_{uuid.uuid4().hex[:10]}.wav"

        try:
            result = self._pipeline(text)
            if isinstance(result, dict):
                audio = result.get("audio") or result.get("array")
                sample_rate = int(result.get("sampling_rate") or result.get("sample_rate") or 24000)
            else:
                audio, sample_rate = result  # type: ignore[misc]
            if audio is None:
                raise ValueError("empty audio from VibeVoice pipeline")
            self._write_wav(audio, sample_rate, out_path)
            warning = lang_warning
            if self._preset.realtime:
                note = f"VibeVoice Realtime ({self.model_id})"
                warning = f"{lang_warning} {note}" if lang_warning else note
            return str(out_path), warning
        except Exception as exc:  # noqa: BLE001
            return self._piper_fallback(
                text,
                language=lang,
                out_dir=out_dir,
                prefix=f"VibeVoice synthesis failed: {exc}. Piper fallback.",
            )

    def _piper_fallback(
        self,
        text: str,
        *,
        language: str,
        out_dir: Path | None,
        prefix: str | None,
    ) -> tuple[str | None, str | None]:
        config = get_echo_coach_config()
        from echocoach.tts.piper import get_tts_backend

        piper = get_tts_backend(config.tts_preset)
        path, piper_warning = piper.synthesize(text, language=language, out_dir=out_dir)
        parts = [p for p in (prefix, piper_warning) if p]
        return path, "; ".join(parts) if parts else None


def get_vibevoice_backend(preset: TtsPreset) -> VibeVoiceTtsBackend:
    if preset.key not in _vibevoice_cache:
        _vibevoice_cache[preset.key] = VibeVoiceTtsBackend(preset)
    return _vibevoice_cache[preset.key]
