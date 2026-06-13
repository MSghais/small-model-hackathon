# gradio-space

Build Small hackathon UI — custom **Studio** frontend plus Classic Gradio tabs.

## Entry points

| URL | What |
|-----|------|
| `/` | **Studio UI** — custom HTML/CSS/JS served via `gradio.Server` |
| `/classic` | Full Gradio Blocks app (all tabs, settings, debug) |

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

- `list_sessions`, `list_documents`
- `ingest_url`, `ingest_files`, `save_upload`
- `generate_slides`
- `teacher_voice_turn`
- `analyze_pitch`
- `model_status`

## Demo script (judges)

1. Open `/` — Photosynthesis project workspace
2. Paste a URL in Research → **Ingest URL** → documents appear with **RAG Active**
3. Center column → **Generate Slides** → slide preview canvas fills
4. Right column → Teacher Voice **Coach** mode → send a question
5. Coach view → upload/record audio → **Analyze pitch** for EchoCoach metrics
6. Fallback: `/classic` for Chat (debug), traces, and model settings

Space card metadata lives in the [repository root README.md](../../README.md).
