"""TTS VoiceOut backends."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

from echocoach.config import TtsPreset, get_echo_coach_config, outputs_dir


class TtsBackend(Protocol):
    def synthesize(self, text: str, *, language: str, out_dir: Path | None = None) -> tuple[str | None, str | None]: ...


_tts_cache: dict[str, "PiperTtsBackend"] = {}


class PiperTtsBackend:
    def __init__(self, preset: TtsPreset) -> None:
        self._preset = preset
        self._voices: dict[str, object] = {}

    def _voice_name(self, language: str) -> tuple[str, str | None]:
        voice = self._preset.voices.get(language)
        warning = None
        if not voice:
            voice = self._preset.fallback_voice
            warning = f"No Piper voice for {language!r}; using {voice}."
        return voice, warning

    def _load_voice(self, voice_name: str):
        if voice_name in self._voices:
            return self._voices[voice_name]
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise ImportError(
                "Piper TTS requires piper-tts. "
                "Install with: uv sync --package echocoach --extra piper"
            ) from exc

        onnx_path = self._resolve_voice_path(voice_name)
        voice = PiperVoice.load(str(onnx_path))
        self._voices[voice_name] = voice
        return voice

    def _voice_search_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        env_dir = __import__("os").environ.get("PIPER_VOICES_DIR")
        if env_dir:
            dirs.append(Path(env_dir))
        dirs.extend(
            [
                Path.home() / ".local" / "share" / "piper" / "voices",
                Path.cwd() / "models" / "piper",
                Path.cwd(),
            ]
        )
        return dirs

    def _resolve_voice_path(self, voice_name: str) -> Path:
        onnx_name = f"{voice_name}.onnx"
        for directory in self._voice_search_dirs():
            path = directory / onnx_name
            if path.is_file():
                return path

        download_targets = [
            Path.cwd() / "models" / "piper",
            Path.home() / ".local" / "share" / "piper" / "voices",
        ]
        try:
            from piper.download_voices import download_voice
        except ImportError:
            download_voice = None

        if download_voice is not None:
            for directory in download_targets:
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    download_voice(voice_name, directory)
                    path = directory / onnx_name
                    if path.is_file():
                        return path
                except Exception:
                    continue

        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "piper.download_voices", voice_name],
            check=True,
        )
        for directory in self._voice_search_dirs():
            path = directory / onnx_name
            if path.is_file():
                return path

        raise FileNotFoundError(
            f"Piper voice {voice_name!r} not found. "
            f"Run: python -m piper.download_voices {voice_name}"
        )

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        out_dir: Path | None = None,
    ) -> tuple[str | None, str | None]:
        if not text.strip():
            return None, "No text to synthesize."

        voice_name, warning = self._voice_name(language)
        try:
            import wave

            from piper import PiperVoice

            voice = self._load_voice(voice_name)
            base = out_dir or outputs_dir()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"voiceout_{uuid.uuid4().hex[:10]}.wav"
            with wave.open(str(out_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            return str(out_path), warning
        except ImportError:
            return None, "piper-tts not installed; VoiceOut skipped."
        except Exception as exc:  # noqa: BLE001
            return None, f"VoiceOut failed: {exc}"


def get_tts_backend(preset_key: str | None = None) -> TtsBackend:
    config = get_echo_coach_config()
    preset = config.get_tts(preset_key)
    if preset.key not in _tts_cache:
        _tts_cache[preset.key] = PiperTtsBackend(preset)
    return _tts_cache[preset.key]
