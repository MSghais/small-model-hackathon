from __future__ import annotations

from pathlib import Path

import pytest

from echocoach.recording import ServerRecordingError, record_server_wav


def test_record_server_wav_uses_arecord_when_sounddevice_missing(tmp_path, monkeypatch):
    out_file = tmp_path / "server_abcd1234.wav"
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: False)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: True)
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
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: False)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: False)

    with pytest.raises(ServerRecordingError, match="No server-side recorder"):
        record_server_wav(max_seconds=5)


def test_record_server_wav_empty_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("echocoach.recording._sounddevice_available", lambda: False)
    monkeypatch.setattr("echocoach.recording._arecord_available", lambda: True)
    monkeypatch.setattr("echocoach.recording.outputs_dir", lambda: tmp_path)

    def empty_arecord(path: Path, seconds: int, sample_rate: int) -> None:
        path.touch()

    monkeypatch.setattr("echocoach.recording._record_arecord", empty_arecord)

    with pytest.raises(ServerRecordingError, match="empty file"):
        record_server_wav(max_seconds=1)
