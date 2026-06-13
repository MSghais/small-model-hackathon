"""Record audio from the server's local microphone (bypasses browser getUserMedia)."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

from echocoach.audio_io import TARGET_SAMPLE_RATE
from echocoach.config import get_echo_coach_config, outputs_dir


class ServerRecordingError(RuntimeError):
    """Raised when server-side capture is unavailable or fails."""


def _sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True


def _arecord_available() -> bool:
    return shutil.which("arecord") is not None


def record_server_wav(
    max_seconds: int | None = None,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Path:
    """Capture mono WAV from this machine's default input device."""
    config = get_echo_coach_config()
    seconds = max_seconds if max_seconds is not None else config.max_seconds
    if seconds <= 0:
        raise ServerRecordingError("Recording duration must be positive.")

    out_dir = outputs_dir() / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"server_{uuid.uuid4().hex[:8]}.wav"

    if _sounddevice_available():
        _record_sounddevice(out_path, seconds, sample_rate)
    elif _arecord_available():
        _record_arecord(out_path, seconds, sample_rate)
    else:
        raise ServerRecordingError(
            "No server-side recorder found. Install sounddevice "
            "(uv sync --package echocoach) or ensure ALSA arecord is on PATH. "
            "You can still upload a .wav file in the browser."
        )

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise ServerRecordingError("Recording finished but produced an empty file.")
    return out_path


def _record_sounddevice(out_path: Path, seconds: int, sample_rate: int) -> None:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    frames = int(seconds * sample_rate)
    try:
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:  # noqa: BLE001 — surface device errors to UI
        raise ServerRecordingError(f"sounddevice capture failed: {exc}") from exc

    sf.write(out_path, np.squeeze(recording), sample_rate)


def _record_arecord(out_path: Path, seconds: int, sample_rate: int) -> None:
    cmd = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-d",
        str(seconds),
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        msg = f"arecord failed (exit {exc.returncode})"
        if detail:
            msg = f"{msg}: {detail}"
        raise ServerRecordingError(msg) from exc
    except OSError as exc:
        raise ServerRecordingError(f"arecord failed: {exc}") from exc
