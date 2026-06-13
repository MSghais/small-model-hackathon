"""Whisper.cpp ASR via pywhispercpp."""

from __future__ import annotations

from echocoach.config import AsrPreset


class WhisperCppBackend:
    def __init__(self, preset: AsrPreset) -> None:
        self._preset = preset
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from pywhispercpp.model import Model
        except ImportError as exc:
            raise ImportError(
                "Whisper.cpp backend requires pywhispercpp. "
                "Install with: uv sync --package echocoach --extra whisper"
            ) from exc

        size = self._preset.model_size or "tiny"
        self._model = Model(size, print_realtime=False, print_progress=False)

    def transcribe(self, audio_path: str, *, language: str) -> str:
        self._load()
        assert self._model is not None

        segments = self._model.transcribe(audio_path, language=language)
        parts = [seg.text.strip() for seg in segments if getattr(seg, "text", "")]
        return " ".join(parts).strip()
