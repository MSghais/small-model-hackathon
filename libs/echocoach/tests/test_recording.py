from __future__ import annotations

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
        self.returncode = 0
        self._out_path.write_bytes(b"RIFFxxxx")

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_select_capture_backend_prefers_pw_record(monkeypatch):
    monkeypatch.setattr("echocoach.recording._pw_record_available", lambda: True)
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: True)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: True)
    assert select_capture_backend() == "pw-record"


def test_start_stop_session(monkeypatch, tmp_path):
    import echocoach.recording as rec

    rec._session = None

    def fake_spawn(backend, path: Path):
        return _FakeProcess(path)

    monkeypatch.setattr("echocoach.recording.select_capture_backend", lambda: "pw-record")
    monkeypatch.setattr("echocoach.recording.outputs_dir", lambda: tmp_path)
    monkeypatch.setattr("echocoach.recording._spawn_capture_process", fake_spawn)
    monkeypatch.setattr("echocoach.recording.threading.Timer", lambda *args, **kwargs: None)

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
    monkeypatch.setattr("echocoach.recording.threading.Timer", lambda *args, **kwargs: None)

    start_server_recording(10)
    with pytest.raises(ServerRecordingError, match="Already recording"):
        start_server_recording(10)
    stop_server_recording()
    rec._session = None


def test_stop_without_start_raises():
    import echocoach.recording as rec

    rec._session = None
    with pytest.raises(ServerRecordingError, match="Not recording"):
        stop_server_recording()


def test_recording_level_warning_detects_silence(tmp_path):
    import numpy as np
    import soundfile as sf

    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16_000, dtype=np.float32), 16_000)
    assert "silent" in (recording_level_warning(silent) or "").lower()
