from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

AsrBackendName = Literal["cohere", "whisper_cpp"]
TtsBackendName = Literal["piper", "vibevoice"]


@dataclass(frozen=True)
class LanguageOption:
    code: str
    label: str


@dataclass(frozen=True)
class AsrPreset:
    key: str
    label: str
    backend: AsrBackendName
    model_id: str | None = None
    model_size: str | None = None


@dataclass(frozen=True)
class TtsPreset:
    key: str
    label: str
    backend: TtsBackendName
    voices: dict[str, str]
    fallback_voice: str
    model_id: str | None = None
    streaming: bool = False
    realtime: bool = False
    supported_languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class EchoCoachConfig:
    asr_preset: str
    tts_preset: str
    realtime_tts_preset: str | None
    coach_model: str
    max_seconds: int
    languages: list[LanguageOption]
    asr_presets: dict[str, AsrPreset]
    tts_presets: dict[str, TtsPreset]
    presets_path: Path | None = None

    def get_asr(self, key: str | None = None) -> AsrPreset:
        preset_key = key or self.asr_preset
        if preset_key not in self.asr_presets:
            known = ", ".join(sorted(self.asr_presets))
            raise KeyError(f"Unknown ASR preset {preset_key!r}. Known: {known}")
        return self.asr_presets[preset_key]

    def get_tts(self, key: str | None = None) -> TtsPreset:
        preset_key = key or self.tts_preset
        if preset_key not in self.tts_presets:
            known = ", ".join(sorted(self.tts_presets))
            raise KeyError(f"Unknown TTS preset {preset_key!r}. Known: {known}")
        return self.tts_presets[preset_key]

    def asr_choices(self) -> list[tuple[str, str]]:
        return [(p.label, p.key) for p in self.asr_presets.values()]

    def language_choices(self) -> list[tuple[str, str]]:
        return [(lang.label, lang.code) for lang in self.languages]


def _find_voice_presets_path() -> Path | None:
    env_path = os.environ.get("VOICE_PRESETS_PATH")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path.resolve()

    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "voice_models.yaml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _builtin_config() -> EchoCoachConfig:
    langs = [
        LanguageOption("en", "English"),
        LanguageOption("fr", "French"),
        LanguageOption("de", "German"),
    ]
    asr = {
        "whisper-cpp-tiny": AsrPreset(
            key="whisper-cpp-tiny",
            label="Whisper.cpp tiny",
            backend="whisper_cpp",
            model_size="tiny",
        ),
    }
    tts = {
        "piper-multilingual": TtsPreset(
            key="piper-multilingual",
            label="Piper TTS",
            backend="piper",
            voices={"en": "en_US-lessac-medium"},
            fallback_voice="en_US-lessac-medium",
        ),
    }
    return EchoCoachConfig(
        asr_preset="whisper-cpp-tiny",
        tts_preset="piper-multilingual",
        realtime_tts_preset=None,
        coach_model="minicpm5-1b",
        max_seconds=30,
        languages=langs,
        asr_presets=asr,
        tts_presets=tts,
    )


def _parse_asr_entry(key: str, raw: dict[str, Any]) -> AsrPreset:
    backend = str(raw.get("backend", "whisper_cpp"))
    if backend not in ("cohere", "whisper_cpp"):
        raise ValueError(f"ASR preset {key!r}: backend must be cohere or whisper_cpp")
    return AsrPreset(
        key=key,
        label=str(raw.get("label", key)),
        backend=backend,  # type: ignore[arg-type]
        model_id=raw.get("model_id"),
        model_size=raw.get("model_size"),
    )


def _parse_tts_entry(key: str, raw: dict[str, Any]) -> TtsPreset:
    backend = str(raw.get("backend", "piper"))
    if backend not in ("piper", "vibevoice"):
        raise ValueError(f"TTS preset {key!r}: backend must be piper or vibevoice")

    voices = raw.get("voices") or {}
    if not isinstance(voices, dict):
        raise ValueError(f"TTS preset {key!r}: voices must be a mapping")
    languages = raw.get("languages") or []
    if backend == "vibevoice":
        if not voices and languages:
            voices = {str(lang): "default" for lang in languages}
        fallback = str(raw.get("fallback_language") or raw.get("fallback_voice") or "en")
    else:
        if not voices:
            raise ValueError(f"TTS preset {key!r}: voices mapping is required")
        fallback = str(raw.get("fallback_voice", "en_US-lessac-medium"))

    return TtsPreset(
        key=key,
        label=str(raw.get("label", key)),
        backend=backend,  # type: ignore[arg-type]
        voices={str(k): str(v) for k, v in voices.items()},
        fallback_voice=fallback,
        model_id=raw.get("model_id"),
        streaming=bool(raw.get("streaming", False)),
        realtime=bool(raw.get("realtime", False)),
        supported_languages=tuple(str(lang) for lang in languages),
    )


def load_echo_coach_config() -> EchoCoachConfig:
    presets_path = _find_voice_presets_path()
    if presets_path is None:
        config = _builtin_config()
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Loading voice_models.yaml requires PyYAML. Install with: uv sync"
            ) from exc

        data = yaml.safe_load(presets_path.read_text()) or {}
        defaults = data.get("defaults", {})
        raw_langs = data.get("languages", [])
        raw_asr = data.get("asr", {})
        raw_tts = data.get("tts", {})

        languages = [
            LanguageOption(code=str(item["code"]), label=str(item["label"]))
            for item in raw_langs
        ]
        asr_presets = {
            key: _parse_asr_entry(key, value) for key, value in raw_asr.items()
        }
        tts_presets = {
            key: _parse_tts_entry(key, value) for key, value in raw_tts.items()
        }

        asr_default = defaults.get("asr_preset", "whisper-cpp-tiny")
        tts_default = defaults.get("tts_preset", "piper-multilingual")
        if asr_default not in asr_presets:
            asr_default = next(iter(asr_presets))
        if tts_default not in tts_presets:
            tts_default = next(iter(tts_presets))

        config = EchoCoachConfig(
            asr_preset=asr_default,
            tts_preset=tts_default,
            realtime_tts_preset=defaults.get("realtime_tts_preset"),
            coach_model=str(defaults.get("coach_model", "minicpm5-1b")),
            max_seconds=int(defaults.get("max_seconds", 30)),
            languages=languages,
            asr_presets=asr_presets,
            tts_presets=tts_presets,
            presets_path=presets_path,
        )

    updates: dict[str, Any] = {}
    if os.environ.get("ECHOCOACH_ASR_PRESET"):
        updates["asr_preset"] = os.environ["ECHOCOACH_ASR_PRESET"]
    if os.environ.get("ECHOCOACH_TTS_PRESET"):
        updates["tts_preset"] = os.environ["ECHOCOACH_TTS_PRESET"]
    if os.environ.get("ECHOCOACH_REALTIME_TTS_PRESET"):
        updates["realtime_tts_preset"] = os.environ["ECHOCOACH_REALTIME_TTS_PRESET"]
    if os.environ.get("ECHOCOACH_COACH_MODEL"):
        updates["coach_model"] = os.environ["ECHOCOACH_COACH_MODEL"]
    if os.environ.get("ECHOCOACH_MAX_SECONDS"):
        updates["max_seconds"] = int(os.environ["ECHOCOACH_MAX_SECONDS"])

    return replace(config, **updates) if updates else config


_config: EchoCoachConfig | None = None


def get_echo_coach_config(reload: bool = False) -> EchoCoachConfig:
    global _config
    if _config is None or reload:
        _config = load_echo_coach_config()
    return _config


def outputs_dir() -> Path:
    env = os.environ.get("AGENT_OUTPUTS_DIR")
    if env:
        return Path(env)
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "voice_models.yaml").is_file() or (base / "pyproject.toml").is_file():
            path = base / "outputs" / "echocoach"
            path.mkdir(parents=True, exist_ok=True)
            return path
    path = Path("/tmp/echocoach_outputs")
    path.mkdir(parents=True, exist_ok=True)
    return path
