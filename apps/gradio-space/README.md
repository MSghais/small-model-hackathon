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

- `teacher_voice_turn`, `teacher_voice_audio_turn`, `teacher_voice_clear`, `teacher_voice_speak`
- `load_sample_pitch`, `analyze_pitch` (language, ASR preset, `speak_rewrite`)
- `recording_status`, `recording_start`, `recording_stop`
- `voice_presets`

**Settings & debug**

- `model_status`, `model_choices`, `reload_model`
- `debug_chat`
- `save_upload`

## Demo script (judges)

1. Open `/` — **Small Model Finetuning** project workspace
2. Paste a URL in Research → **Ingest URL** → documents appear with **RAG Active**
3. Center column → **Generate Slides** → slide preview canvas, thumbnail strip, and **Outline** panel
4. Optional: expand **Research sources** → Web search or RAG modes
5. Voice view → text or **mic** → full conversation thread + **Speak full reply**
6. Coach view → **Load sample clip** or record → **Analyze pitch** (charts, transcript, VoiceOut)
7. Debug sidebar → RAG scope overrides, plain chat or corpus-grounded test with traces
8. Settings drawer → model status / reload (Classic at `/classic` still available)

Space card metadata lives in the [repository root README.md](../../README.md).
