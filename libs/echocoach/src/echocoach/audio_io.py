from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000


def load_audio_mono_16k(path: str | Path) -> tuple[np.ndarray, float]:
    """Load audio as mono float32 at 16 kHz; return (samples, duration_seconds)."""
    import librosa

    audio, _ = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    duration = len(audio) / TARGET_SAMPLE_RATE if len(audio) else 0.0
    return audio.astype(np.float32), duration


def clamp_duration(audio: np.ndarray, max_seconds: float) -> np.ndarray:
    max_samples = int(max_seconds * TARGET_SAMPLE_RATE)
    if len(audio) > max_samples:
        return audio[:max_samples]
    return audio


def write_wav_temp(audio: np.ndarray, directory: Path, stem: str = "clip") -> Path:
    import soundfile as sf

    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{stem}.wav"
    sf.write(out, audio, TARGET_SAMPLE_RATE)
    return out


def count_words(text: str) -> int:
    tokens = re.findall(r"\b[\w']+\b", text, flags=re.UNICODE)
    return len(tokens)
