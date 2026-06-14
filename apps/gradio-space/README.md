# gradio-space

Build Small hackathon UI — custom **Studio** frontend plus Classic Gradio tabs.

## Entry points

| URL | What |
|-----|------|
| `/` | **Studio UI** — custom HTML/CSS/JS served via `gradio.Server` (near parity with Classic) |
| `/classic` | Full Gradio Blocks app (fallback / power-user tabs) |

```bash
uv run --package gradio-space python -m gradio_space.server
# or
uv run --package gradio-space python -m gradio_space.app
```

## Off Brand architecture

This package uses **Gradio 6 Server mode** (`gradio.Server`):

- Custom routes: `GET /`, static assets at `/static/studio/`
- API endpoints via `@server.api(name=...)` — callable from `@gradio/client` and `gradio_client`
- Classic UI mounted with `mount_gradio_app(..., path="/classic")`

### Studio API names

**Research & slides**

- `list_sessions`, `list_documents`, `session_memory`
- `discover_sources`, `auto_search_ingest`, `ingest_sources`, `ingest_url`, `ingest_files`
- `research_chat`, `generate_slides` (supports `source_mode`: none / web / rag)

**Voice & coach**

- `language_lesson_turn` — unified text/audio turn for Language lessons (mode, language, `auto_voiceout`, coach variant)
- `teacher_voice_turn`, `teacher_voice_audio_turn`, `teacher_voice_clear`, `teacher_voice_speak`
- `load_sample_pitch`, `analyze_pitch` (Classic EchoCoach; language, ASR preset, `speak_rewrite`)
- `recording_status`, `recording_start`, `recording_stop`
- `voice_presets`

**Settings & debug**

- `model_status`, `model_choices`, `reload_model`
- `debug_chat`
- `save_upload`

## Demo script (judges) — Language lessons + Cohere stack

**Badge line:** Cohere Labs — Transcribe + Tiny Aya on one local Language lessons page.

1. Open `/` — **Small Model Finetuning** project workspace
2. **Language lessons** tab → select **French** → hold mic → ask *« Explique le fine-tuning en termes simples. »* → hear Piper/VibeVoice reply
3. Switch to **Spanish**, type a follow-up (text in, text + audio out with **Auto-speak replies** on)
4. Select **Other (text only)** → enter `hi` → show Tiny Aya Fire-quality written lesson (text only banner)
5. Toggle **Use indexed sources** after ingesting one PDF in **Research**
6. Optional: **Generate Slides** from the Slides tab; **Classic UI** (`/classic`) for EchoCoach pitch metrics

Space secrets for GPU demo:

```bash
ECHOCOACH_ASR_PRESET=cohere-transcribe
ECHOCOACH_COACH_MODEL=tiny-aya-global
ECHOCOACH_REALTIME_TTS_PRESET=vibevoice-realtime-0.5b
```

Space card metadata lives in the [repository root README.md](../../README.md).
