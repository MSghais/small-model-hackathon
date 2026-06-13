"""Record audio from the server's local microphone (bypasses browser getUserMedia)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from echocoach.audio_io import TARGET_SAMPLE_RATE, load_audio_mono_16k
from echocoach.config import get_echo_coach_config, outputs_dir

CaptureBackend = Literal["pw-record", "sounddevice", "arecord"]
SILENT_RMS_THRESHOLD = 0.002


class ServerRecordingError(RuntimeError):
    """Raised when server-side capture is unavailable or fails."""


@dataclass
class _RecordingSession:
    process: subprocess.Popen[bytes]
    out_path: Path
    backend: CaptureBackend
    max_seconds: int
    started_at: float
    watchdog: threading.Timer | None = None


_session: _RecordingSession | None = None
_session_lock = threading.Lock()


def _capture_device() -> str | None:
    device = os.environ.get("ECHOCOACH_CAPTURE_DEVICE", "").strip()
    return device or None


def _pw_record_available() -> bool:
    return shutil.which("pw-record") is not None


def _sounddevice_available() -> bool:
    try:
        import sounddevice as sd  # noqa: PLC0415
    except (ImportError, OSError):
        return False
    try:
        sd.query_devices()
    except Exception:
        return False
    return True


def _arecord_available() -> bool:
    if shutil.which("arecord") is None:
        return False
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = f"{result.stdout}\n{result.stderr}".lower()
    return "card" in output


def select_capture_backend() -> CaptureBackend | None:
    """Pick the best capture tool for this machine."""
    if _pw_record_available():
        return "pw-record"
    if _sounddevice_available():
        return "sounddevice"
    if _arecord_available():
        return "arecord"
    return None


def recording_backend_status() -> str:
    backend = select_capture_backend()
    device = _capture_device()
    if backend == "pw-record":
        note = "PipeWire pw-record"
    elif backend == "sounddevice":
        note = "sounddevice / PortAudio"
    elif backend == "arecord":
        note = "ALSA arecord"
    else:
        note = None

    if note:
        extra = f" (device: `{device}`)" if device else ""
        return f"Server microphone: ready ({note}{extra}). Click **Start recording**, speak, then **Stop recording**."

    hints: list[str] = []
    if not _pw_record_available() and not _arecord_available():
        hints.append("install PipeWire (`pw-record`) or ALSA utils (`arecord`)")
    elif not _arecord_available():
        hints.append("enable a microphone in system sound settings")
    if not _sounddevice_available():
        hints.append(
            "optional: `sudo apt install libportaudio2` for sounddevice fallback"
        )
    hint = "; ".join(hints) if hints else "no capture backend available"
    return (
        f"Server microphone: unavailable — {hint}. "
        "Use **Upload** or open **http://localhost:7860** in Chrome/Firefox for browser mic."
    )


def is_recording_active() -> bool:
    with _session_lock:
        return _session is not None and _session.process.poll() is None


def recording_elapsed_seconds() -> float:
    with _session_lock:
        if _session is None:
            return 0.0
        return max(0.0, time.monotonic() - _session.started_at)


def start_server_recording(max_seconds: int | None = None) -> None:
    """Begin an open-ended capture; call stop_server_recording() to finish."""
    config = get_echo_coach_config()
    seconds = max_seconds if max_seconds is not None else config.max_seconds
    if seconds <= 0:
        raise ServerRecordingError("Recording duration must be positive.")

    backend = select_capture_backend()
    if backend is None:
        raise ServerRecordingError(recording_backend_status())

    out_dir = outputs_dir() / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"server_{uuid.uuid4().hex[:8]}.wav"

    with _session_lock:
        if _session is not None and _session.process.poll() is None:
            raise ServerRecordingError("Already recording. Click **Stop recording** first.")

        process = _spawn_capture_process(backend, out_path)
        watchdog = threading.Timer(seconds, _auto_stop_recording)
        watchdog.daemon = True
        watchdog.start()
        _session = _RecordingSession(
            process=process,
            out_path=out_path,
            backend=backend,
            max_seconds=seconds,
            started_at=time.monotonic(),
            watchdog=watchdog,
        )


def stop_server_recording() -> Path:
    """Stop the active capture and return the WAV path."""
    with _session_lock:
        session = _session
        if session is None or session.process.poll() is not None:
            raise ServerRecordingError("Not recording. Click **Start recording** first.")
        return _finalize_session(session)


def _auto_stop_recording() -> None:
    with _session_lock:
        session = _session
        if session is None or session.process.poll() is not None:
            return
        try:
            _finalize_session(session)
        except ServerRecordingError:
            pass


def _finalize_session(session: _RecordingSession) -> Path:
    global _session

    if session.watchdog is not None:
        session.watchdog.cancel()

    process = session.process
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    if process.returncode not in (0, -signal.SIGINT, 128 + signal.SIGINT):
        detail = ""
        if process.stderr:
            detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        msg = f"Capture stopped with exit code {process.returncode}"
        if detail:
            msg = f"{msg}: {detail}"
        _session = None
        raise ServerRecordingError(msg)

    out_path = session.out_path
    if not out_path.is_file() or out_path.stat().st_size == 0:
        _session = None
        raise ServerRecordingError("Recording finished but produced an empty file.")

    _session = None
    return out_path


def analyze_recording_levels(path: str | Path) -> tuple[float, float, float]:
    audio, duration = load_audio_mono_16k(path)
    if len(audio) == 0:
        return 0.0, 0.0, duration
    import numpy as np

    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return rms, peak, duration


def recording_level_warning(path: str | Path) -> str | None:
    rms, peak, duration = analyze_recording_levels(path)
    if duration < 0.2:
        return "Clip is very short — try recording a bit longer."
    if rms < SILENT_RMS_THRESHOLD and peak < SILENT_RMS_THRESHOLD * 4:
        return (
            "Recording looks silent. Check system mic input/mute, pick the right input device "
            "(set `ECHOCOACH_CAPTURE_DEVICE`), or use **Upload**."
        )
    return None


def record_server_wav(
    max_seconds: int | None = None,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Path:
    """Fixed-length capture (used in tests and scripts)."""
    start_server_recording(max_seconds)
    time.sleep(max_seconds if max_seconds is not None else get_echo_coach_config().max_seconds)
    return stop_server_recording()


def _spawn_capture_process(backend: CaptureBackend, out_path: Path) -> subprocess.Popen[bytes]:
    device = _capture_device()
    if backend == "pw-record":
        cmd = [
            "pw-record",
            "--media-category",
            "Capture",
            "--media-role",
            "Speech",
            "--rate",
            str(TARGET_SAMPLE_RATE),
            "--channels",
            "1",
            "--format",
            "s16",
        ]
        if device:
            cmd.extend(["--target", device])
        cmd.append(str(out_path))
    elif backend == "arecord":
        cmd = [
            "arecord",
            "-q",
            "-f",
            "S16_LE",
            "-r",
            str(TARGET_SAMPLE_RATE),
            "-c",
            "1",
        ]
        if device:
            cmd.extend(["-D", device])
        else:
            cmd.extend(["-D", "pipewire"])
        cmd.append(str(out_path))
    else:
        raise ServerRecordingError(
            "sounddevice does not support open-ended capture yet; install `pw-record` or use arecord."
        )

    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        raise ServerRecordingError(f"Failed to start {backend}: {exc}") from exc


def _record_sounddevice(out_path: Path, seconds: int, sample_rate: int) -> None:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    frames = int(seconds * sample_rate)
    device = _capture_device()
    try:
        recording = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as exc:  # noqa: BLE001 — surface device errors to UI
        raise ServerRecordingError(f"sounddevice capture failed: {exc}") from exc

    sf.write(out_path, np.squeeze(recording), sample_rate)
