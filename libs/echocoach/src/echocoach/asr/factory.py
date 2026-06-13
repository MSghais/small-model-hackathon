"""ASR backend protocol and factory."""

from __future__ import annotations

from typing import Protocol

from echocoach.config import AsrPreset, get_echo_coach_config


class AsrBackend(Protocol):
    def transcribe(self, audio_path: str, *, language: str) -> str: ...


_asr_cache: dict[tuple, AsrBackend] = {}


def get_asr_backend(preset_key: str | None = None) -> AsrBackend:
    config = get_echo_coach_config()
    preset = config.get_asr(preset_key)
    cache_key = (preset.key, preset.backend, preset.model_id, preset.model_size)
    if cache_key in _asr_cache:
        return _asr_cache[cache_key]

    if preset.backend == "cohere":
        from echocoach.asr.cohere import CohereAsrBackend

        backend: AsrBackend = CohereAsrBackend(preset)
    elif preset.backend == "whisper_cpp":
        from echocoach.asr.whisper_cpp import WhisperCppBackend

        backend = WhisperCppBackend(preset)
    else:
        raise ValueError(f"Unsupported ASR backend: {preset.backend}")

    _asr_cache[cache_key] = backend
    return backend


def reset_asr_backends() -> None:
    _asr_cache.clear()
