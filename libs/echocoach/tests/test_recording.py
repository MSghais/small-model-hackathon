from __future__ import annotations

from pathlib import Path

import pytest

from echocoach.recording import (
    ServerRecordingError,
    recording_level_warning,
    select_capture_backend,
    start_server_recording,
    stop_server_recording,
)


class _FakeProcess:
    def __init__(self, out_path: Path) -> None:
        self._running = True
        self.returncode: int | None = None
        self.stderr = None
        self._out_path = out_path

    def poll(self) -> int | None:
        return None if self._running else self.returncode

    def send_signal(self, sig: int) -> None:
        self._running = False
        self.returncode = 1  # pw-record exits 1 on SIGINT
        self._out_path.write_bytes(b"RIFF" + b"x" * 100)

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_select_capture_backend_prefers_pw_record(monkeypatch):
    monkeypatch.setattr("echocoach.recording._pw_record_available", lambda: True)
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: True)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: True)
    assert select_capture_backend() == "pw-record"


class _FakeTimer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.daemon = False

    def start(self) -> None:
        return None

    def cancel(self) -> None:
        return None


def test_start_stop_session(monkeypatch, tmp_path):
    import echocoach.recording as rec

    rec._session = None

    def fake_spawn(backend, path: Path):
        return _FakeProcess(path)

    monkeypatch.setattr("echocoach.recording.select_capture_backend", lambda: "pw-record")
    monkeypatch.setattr("echocoach.recording.outputs_dir", lambda: tmp_path)
    monkeypatch.setattr("echocoach.recording._spawn_capture_process", fake_spawn)
    monkeypatch.setattr("echocoach.recording.threading.Timer", _FakeTimer)

    start_server_recording(10)
    path = stop_server_recording()
    assert path.is_file()
    rec._session = None


def test_start_while_recording_raises(monkeypatch, tmp_path):
    import echocoach.recording as rec

    rec._session = None

    def fake_spawn(backend, path: Path):
        return _FakeProcess(path)

    monkeypatch.setattr("echocoach.recording.select_capture_backend", lambda: "pw-record")
    monkeypatch.setattr("echocoach.recording.outputs_dir", lambda: tmp_path)
    monkeypatch.setattr("echocoach.recording._spawn_capture_process", fake_spawn)
    monkeypatch.setattr("echocoach.recording.threading.Timer", _FakeTimer)

    start_server_recording(10)
    with pytest.raises(ServerRecordingError, match="Already recording"):
        start_server_recording(10)
    stop_server_recording()
    rec._session = None


def test_stop_without_start_raises():
    import echocoach.recording as rec

    rec._session = None
    rec._last_recording_path = None
    with pytest.raises(ServerRecordingError, match="Not recording"):
        stop_server_recording()


def test_recording_level_warning_detects_silence(tmp_path):
    import numpy as np
    import soundfile as sf

    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16_000, dtype=np.float32), 16_000)
    assert "silent" in (recording_level_warning(silent) or "").lower()


def test_finalize_accepts_pw_record_exit_code_one(tmp_path, monkeypatch):
    import echocoach.recording as rec

    rec._session = None
    rec._last_recording_path = None

    out_path = tmp_path / "recordings" / "server_pw.wav"
    out_path.parent.mkdir(parents=True)
    fake_proc = _FakeProcess(out_path)

    session = rec._RecordingSession(
        process=fake_proc,
        out_path=out_path,
        backend="pw-record",
        max_seconds=10,
        started_at=0.0,
        watchdog=None,
    )
    rec._session = session
    path = rec.stop_server_recording()
    assert path == out_path
    assert path.stat().st_size > 44
    rec._session = None
