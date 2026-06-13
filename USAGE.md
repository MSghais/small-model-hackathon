# Usage

How to run the **Lesson Agent** Gradio app locally, test it in Docker, and deploy to a Hugging Face Space for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).

The primary UI is the **Lesson slides** tab (topic → local model outline → downloadable `.pptx`). Use **ResearchMind** for corpus Q&A, **TeacherVoice** for spoken back-and-forth tutoring, **EchoCoach** for one-shot pitch analysis, or ground lessons directly from the Lesson tab. The **Chat (debug)** tab tests the underlying model.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Python 3.12 (see `.python-version`)
- For Docker testing: Docker installed locally
- For HF Space deploy: Hugging Face account with access to the `build-small-hackathon` org

## Local development

### 1. Install dependencies

```bash
uv sync --all-packages
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
```

Edit `.env` if you want a different model preset. Default is `minicpm5-1b` (transformers).

### 3. Pre-download the model (optional for GGUF presets)

If using a GGUF preset (`qwen3b-gguf`), pre-download avoids a long wait on first use:

```bash
uv run python scripts/download_model.py
```

Then add the printed path to `.env`:

```bash
MODEL_PATH=./models/qwen2.5-3b-instruct-q4_k_m.gguf
```

### 4. Run the Gradio app

```bash
uv run --package gradio-space python -m gradio_space.app
```

Open [http://localhost:7860](http://localhost:7860).

| URL | UI |
|-----|-----|
| `/` | **Studio** — custom HTML/CSS/JS workspace (Off Brand entry) |
| `/classic` | **Classic** — full Gradio tabs, settings, Chat (debug) |

The header in Classic includes a link back to Studio UI.

The model loads on the **first Generate** (Lesson slides) or chat message. Agent traces are written to `outputs/traces/`. After code changes, restart the process to pick up updates.

### Lesson slides — research sources

The **Lesson slides** tab can ground outlines on external sources before building the deck:

| Source mode | What it does |
| ----------- | ------------ |
| **None (model only)** | Default — outline from the local model only |
| **Web search** | Search the web for the lesson topic, ingest pages, retrieve passages, then draft slides |
| **RAG (indexed sources)** | Use a **ResearchMind session** and/or URLs/files you provide on this tab |

When **Web search** is selected, choose a **search workflow**:

| Workflow | Steps |
| -------- | ----- |
| **Two-step search (suggest & confirm)** | Click **Discover sources** → select URLs → **Generate lesson slides** |
| **Auto search & ingest** | Click **Generate lesson slides** only — search, ingest, and outline in one step |

**RAG** mode accepts an optional ResearchMind session, document checkboxes (scope), pasted URLs, and PDF/DOCX uploads. Indexed content is retrieved and passed to the outline step.

Web discover/auto search requires network access. MemRAG data is stored under `RESEARCHMIND_DATA_DIR` (default `outputs/researchmind`).

Web discover/auto search requires network access. MemRAG data is stored under `RESEARCHMIND_DATA_DIR` (default `outputs/researchmind`).

### EchoCoach — voice practice

The **EchoCoach** tab records up to 30 seconds, then runs a local pipeline:

**Getting audio in**

- **Record from this computer** — click **Start recording**, speak, then **Stop recording** (uses PipeWire `pw-record` when available). The slider is a max-length safety cap.
- **Browser Record** — needs mic permission and a secure context; open **http://localhost:7860** (not `0.0.0.0` or a LAN IP).
- **Upload** — drop a `.wav` or `.mp3` file (works everywhere, including HF Space).

If recordings sound silent, check system mic input/mute or set `ECHOCOACH_CAPTURE_DEVICE` in `.env` (see `arecord -L` or `pw-cli ls Node`).

Pipeline steps:

1. **ASR** — Cohere Transcribe 2B (14 languages) or Whisper.cpp tiny/base
2. **Analysis** — filler highlights, pace score, matplotlib charts
3. **Coach** — rewrite + tips from the text LLM (`ACTIVE_MODEL`, default `minicpm5-1b`)
4. **VoiceOut** — Piper TTS speaks the summary (or full rewrite if checked)

Install optional extras:

```bash
# Whisper.cpp fallback ASR (CPU)
uv sync --package echocoach --extra whisper

# Piper VoiceOut TTS
uv sync --package echocoach --extra piper
python -m piper.download_voices en_US-lessac-medium
```

Configure presets in [`voice_models.yaml`](voice_models.yaml) or via `.env`:

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `ECHOCOACH_ASR_PRESET` | `whisper-cpp-tiny` | ASR preset key |
| `ECHOCOACH_TTS_PRESET` | `piper-multilingual` | TTS preset key (EchoCoach, default VoiceOut) |
| `ECHOCOACH_REALTIME_TTS_PRESET` | `vibevoice-realtime-0.5b` | TeacherVoice streaming TTS (see below) |
| `ECHOCOACH_COACH_MODEL` | `minicpm5-1b` | Text coach preset (from `models.yaml`) |
| `ECHOCOACH_MAX_SECONDS` | `30` | Max recording length |

**Cohere Transcribe** (`cohere-transcribe`) is gated on Hugging Face — run `huggingface-cli login`, accept the model terms, then set `ECHOCOACH_ASR_PRESET=cohere-transcribe`. GPU recommended for ASR + coach together.

Smoke tests (analysis only, no GPU):

```bash
bash scripts/echo_coach_smoke.sh
```

### TeacherVoice — spoken conversation (turn-based)

The **TeacherVoice** tab is a **multi-turn voice teacher** — not full duplex like a phone call, but speak → wait → hear a reply → repeat.

| Mode | Purpose |
| ---- | ------- |
| **Explain** | Tutor any topic in plain language |
| **Lesson coach** | Discuss and outline lesson content verbally |
| **Pitch practice** | Short live speaking tips each turn |

**EchoCoach vs TeacherVoice**

| | EchoCoach | TeacherVoice |
| --- | --- | --- |
| Interaction | One-shot after **Analyze pitch** | Multi-turn **Send turn** |
| Best for | Pace/filler charts, JSON rewrite report | Q&A, lesson discussion, conversational pitch tips |
| TTS | One VoiceOut clip per analysis | Voice reply every turn (first sentence plays quickly when Piper is installed) |
| RAG | No | Optional ResearchMind grounding (Explain / Lesson) |

**Flow per turn:** record up to **15s** → ASR → text LLM with chat history → Piper TTS (auto-plays when installed).

After each reply, use **Speak last reply** or **Speak first sentence** to generate or replay VoiceOut from the latest assistant message (works even if auto-TTS was skipped).

Install Piper for voice output (included in `gradio-space` deps after `uv sync`):

```bash
uv sync
python -m piper.download_voices en_US-lessac-medium
```

Voices are stored under `models/piper/` (gitignored) or `~/.local/share/piper/voices/`. **Restart the Gradio app** after installing Piper so the Speak buttons can synthesize audio.

**Realtime TTS (VibeVoice)** — [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) is registered in `voice_models.yaml` as `vibevoice-realtime-0.5b` (~300 ms to first audio, streaming text-in). TeacherVoice uses `realtime_tts_preset` from YAML by default; override with `ECHOCOACH_REALTIME_TTS_PRESET` or set `ECHOCOACH_TTS_PRESET=vibevoice-realtime-0.5b` globally. GPU recommended; falls back to Piper until the model loads. English-first; de/fr/it/es/pt/nl/pl/ja/ko are experimental per the model card.

Enable RAG in the accordion: pick a ResearchMind session and optional documents (same scope rules as Chat debug).

Reuse VoiceOut in other tabs via `gradio_space.voice_helpers.speak_last_assistant_reply`.

Optional omni profile (GPU, experimental — falls back to ASR+LLM+Piper):

```bash
ECHOCOACH_VOICE_PROFILE=omni
ECHOCOACH_OMNI_MODEL=openbmb/MiniCPM-o-4_5
```

Unit tests (no GPU):

```bash
uv run pytest libs/echocoach/tests/test_teacher_voice.py -q
```

### 5. Upload agent trace (Sharing is Caring badge)

```bash
uv run python scripts/upload_trace.py --repo-id YOUR_USER/build-small-agent-traces
```

### 5. Quick sanity checks

```bash
# Inference package resolves
uv run python -c "from inference.factory import get_backend; print(type(get_backend()).__name__)"

# Gradio app module loads
uv run --package gradio-space python -c "from gradio_space.app import build_demo; print(build_demo())"
```

### Local env reference


| Variable            | Default                           | Description                                |
| ------------------- | --------------------------------- | ------------------------------------------ |
| `INFERENCE_BACKEND` | `llama_cpp`                       | `llama_cpp` or `transformers`              |
| `MODEL_REPO`        | `Qwen/Qwen2.5-3B-Instruct-GGUF`   | Hub repo for GGUF                          |
| `MODEL_FILE`        | `qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF filename                              |
| `MODEL_PATH`        | —                                 | Local GGUF path (skips Hub download)       |
| `N_CTX`             | `4096`                            | Context window                             |
| `N_GPU_LAYERS`      | `0`                               | GPU layers for llama.cpp (`0` = CPU only)  |
| `PORT`              | `7860`                            | Gradio listen port                         |
| `MODEL_ID`          | `Qwen/Qwen2.5-3B-Instruct`        | Used when `INFERENCE_BACKEND=transformers` |


### Optional: transformers backend

Heavier install; only needed if you switch away from llama.cpp:

```bash
uv sync --package inference --extra transformers
INFERENCE_BACKEND=transformers MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
  uv run --package gradio-space python -m gradio_space.app
```

---

## Docker (local prod-like test)

Run the same container image HF Spaces will build:

```bash
docker build -t hackathon-space .
docker run --rm -p 7860:7860 \
  -e MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF \
  -e MODEL_FILE=qwen2.5-3b-instruct-q4_k_m.gguf \
  -e N_CTX=4096 \
  -e N_GPU_LAYERS=0 \
  hackathon-space
```

Open [http://localhost:7860](http://localhost:7860) — Studio at `/`, Classic tabs at `/classic`. Stop with `Ctrl+C`.

To use a pre-downloaded local model inside Docker, mount it and set `MODEL_PATH`:

```bash
docker run --rm -p 7860:7860 \
  -v "$(pwd)/models:/app/models:ro" \
  -e MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  hackathon-space
```

---

## Hugging Face Space deployment

This repo uses the **Docker SDK**. The Space card metadata lives in the YAML frontmatter at the top of [README.md](README.md).

### 1. Push code to GitHub

Make sure `main` (or your deploy branch) contains at minimum:

- `Dockerfile`
- `README.md` (with `sdk: docker` and `app_port: 7860`)
- `pyproject.toml`, `uv.lock`
- `apps/gradio-space/` and `libs/inference/`

### 2. Create the Space

1. Go to [build-small-hackathon](https://huggingface.co/build-small-hackathon)
2. **New Space**
3. Name: e.g. `small-model-hackathon`
4. SDK: **Docker**
5. Link your GitHub repo, or push directly to the Space repo

CLI alternative (if you have `hf` installed and org access):

```bash
hf repo create build-small-hackathon/<your-space-name> \
  --repo-type space \
  --space_sdk docker
```

### 3. Configure hardware


| Setting  | Recommendation                                               |
| -------- | ------------------------------------------------------------ |
| Hardware | **CPU basic** to start (llama.cpp with `N_GPU_LAYERS=0`)     |
| Upgrade  | GPU Space if you set `N_GPU_LAYERS > 0` for faster inference |


### 4. Set Space environment variables

In the Space **Settings → Variables and secrets**:


| Variable            | Value                             |
| ------------------- | --------------------------------- |
| `INFERENCE_BACKEND` | `llama_cpp`                       |
| `MODEL_REPO`        | `Qwen/Qwen2.5-3B-Instruct-GGUF`   |
| `MODEL_FILE`        | `qwen2.5-3b-instruct-q4_k_m.gguf` |
| `N_CTX`             | `4096`                            |
| `N_GPU_LAYERS`      | `0` (or higher on GPU hardware)   |


### 5. Build and verify

HF builds from the root `Dockerfile` and runs:

```bash
uv run --package gradio-space python -m gradio_space.app
```

Check the **Logs** tab while the Space builds. Once running, open the Space URL and send a test chat message. The first message may take several minutes on CPU while the GGUF downloads.

### 6. Optional: persistent model cache

If cold starts are too slow, attach a **Storage Bucket** in Space settings so downloaded GGUF files survive restarts.

---

## Troubleshooting


| Symptom                                  | Likely cause                      | Fix                                                                  |
| ---------------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| First chat hangs / slow                  | GGUF downloading from Hub         | Pre-download locally; on Space, wait or use Storage Bucket           |
| `Failed to load model` in chat           | Wrong `MODEL_REPO` / `MODEL_FILE` | Check env vars match a valid GGUF on Hub                             |
| Docker build fails on `llama-cpp-python` | Missing build tools               | Dockerfile already installs `build-essential` and `cmake`            |
| Space build fails                        | Missing `uv.lock` or README YAML  | Ensure `sdk: docker` is in root `README.md` frontmatter              |
| `transformers` backend error             | Optional deps not installed       | Run `uv sync --package inference --extra transformers`               |
| Port already in use locally              | Another process on 7860           | `PORT=7861 uv run --package gradio-space python -m gradio_space.app` |


---

## Entrypoint summary

All three environments use the same command:

```bash
uv run --package gradio-space python -m gradio_space.app
```


| Environment | How to run                                                 |
| ----------- | ---------------------------------------------------------- |
| Local dev   | `uv run --package gradio-space python -m gradio_space.app` |
| Docker      | `docker run -p 7860:7860 hackathon-space`                  |
| HF Space    | Built and started automatically from `Dockerfile` `CMD`    |


