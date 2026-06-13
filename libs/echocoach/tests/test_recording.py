from __future__ import annotations

from pathlib import Path

import pytest

from echocoach.recording import (
    ServerRecordingError,
    record_server_wav,
    select_recording_backend,
)


def test_select_recording_backend_prefers_sounddevice(monkeypatch):
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: True)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: True)
    assert select_recording_backend() == "sounddevice"


def test_select_recording_backend_falls_back_to_arecord(monkeypatch):
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: False)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: True)
    assert select_recording_backend() == "arecord"


def test_record_server_wav_uses_arecord_when_sounddevice_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: False)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: True)
    monkeypatch.setattr("echocoach.recording.select_recording_backend", lambda: "arecord")
    monkeypatch.setattr(
        "echocoach.recording.outputs_dir",
        lambda: tmp_path,
    )

    def fake_arecord(path: Path, seconds: int, sample_rate: int) -> None:
        assert seconds == 30
        assert sample_rate == 16_000
        path.write_bytes(b"RIFF")

    monkeypatch.setattr("echocoach.recording._record_arecord", fake_arecord)

    path = record_server_wav()
    assert path.parent == tmp_path / "recordings"
    assert path.name.startswith("server_")


def test_record_server_wav_raises_when_no_backend(monkeypatch):
    monkeypatch.setattr("echocoach.recording.select_recording_backend", lambda: None)
    monkeypatch.setattr(
        "echocoach.recording.recording_backend_status",
        lambda: "Server microphone: unavailable",
    )

    with pytest.raises(ServerRecordingError, match="unavailable"):
        record_server_wav(max_seconds=5)


def test_record_server_wav_empty_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("echocoach.recording.select_recording_backend", lambda: "arecord")
    monkeypatch.setattr("echocoach.recording.outputs_dir", lambda: tmp_path)

    def empty_arecord(path: Path, seconds: int, sample_rate: int) -> None:
        path.touch()

    monkeypatch.setattr("echocoach.recording._record_arecord", empty_arecord)

    with pytest.raises(ServerRecordingError, match="empty file"):
        record_server_wav(max_seconds=1)
