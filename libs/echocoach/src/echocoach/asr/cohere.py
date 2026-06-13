"""Cohere Transcribe 2B ASR backend."""

from __future__ import annotations

from echocoach.audio_io import TARGET_SAMPLE_RATE, load_audio_mono_16k
from echocoach.config import AsrPreset


class CohereAsrBackend:
    def __init__(self, preset: AsrPreset) -> None:
        self._preset = preset
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        model_id = self._preset.model_id or "CohereLabs/cohere-transcribe-03-2026"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
        )

    def transcribe(self, audio_path: str, *, language: str) -> str:
        self._load()
        assert self._processor is not None
        assert self._model is not None

        audio, _ = load_audio_mono_16k(audio_path)
        inputs = self._processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            language=language,
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)

        outputs = self._model.generate(**inputs, max_new_tokens=512)
        decoded = self._processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
        if isinstance(decoded, list):
            return decoded[0].strip() if decoded else ""
        return str(decoded).strip()
