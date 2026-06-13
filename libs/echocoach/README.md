# echocoach

Local voice practice coach for the Build Small Hackathon.

- **ASR:** Cohere Transcribe 2B or Whisper.cpp (tiny/base)
- **Analysis:** filler detection, pace scoring, matplotlib charts
- **Coach:** text LLM via `inference` (default `minicpm5-1b`)
- **VoiceOut:** Piper TTS (optional extra)

Configure presets in repo-root `voice_models.yaml`.
