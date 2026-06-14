# Usage

How to run the **Lesson Agent** Gradio app locally, deploy to a Hugging Face Space (Gradio SDK + ZeroGPU), and optionally test with Docker later for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).

The primary UI is the **Lesson slides** tab (topic → local model outline → downloadable `.pptx`). Use **ResearchMind** for corpus Q&A, **Language lessons** for multilingual text + voice tutoring (Cohere Transcribe + Tiny Aya), **EchoCoach** for one-shot pitch analysis in Classic UI, or ground lessons directly from the Lesson tab. The **Chat (debug)** tab tests the underlying model.

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
| `ECHOCOACH_ASR_PRESET` | `cohere-transcribe` | ASR preset key (Space demo); use `whisper-cpp-tiny` on CPU dev |
| `ECHOCOACH_TTS_PRESET` | `piper-multilingual` | TTS preset key (EchoCoach, default VoiceOut) |
| `ECHOCOACH_REALTIME_TTS_PRESET` | `vibevoice-realtime-0.5b` | Language lessons streaming TTS (see below) |
| `ECHOCOACH_COACH_MODEL` | `tiny-aya-global` | Text coach preset (Tiny Aya; from `models.yaml`) |
| `ECHOCOACH_COACH_FALLBACK` | `minicpm5-1b` | Comma-separated fallback presets if primary coach fails to load |
| `ECHOCOACH_MAX_SECONDS` | `30` | Max recording length |

**Cohere Transcribe** (`cohere-transcribe`) is gated on Hugging Face — run `huggingface-cli login`, accept the model terms, then set `ECHOCOACH_ASR_PRESET=cohere-transcribe`. GPU recommended for ASR + coach together.

Smoke tests (analysis only, no GPU):

```bash
bash scripts/echo_coach_smoke.sh
```

### Language lessons — multilingual coach (Studio tab)

The **Language lessons** tab is the primary voice learning experience: one page for **text**, **hold-to-talk mic**, and **audio upload**, with optional auto VoiceOut on every reply.

| Input | Output |
| ----- | ------ |
| Type a question | Chat bubble in target language |
| Hold mic / upload audio | Transcript + teacher reply; auto-play TTS when enabled |
| **Other (text only)** language code | Tiny Aya written lesson (no Piper voice for unsupported codes) |

**Stack (Cohere Labs partner demo):** [Cohere Transcribe](https://huggingface.co/CohereLabs/c4ai-transcribe-v2) (14 voice langs) → [Tiny Aya Global / regional](https://huggingface.co/CohereLabs/tiny-aya-global) (70+ text langs) → Piper or VibeVoice Realtime for speech out.

Set Space secrets (GPU recommended):

```bash
ECHOCOACH_ASR_PRESET=cohere-transcribe
ECHOCOACH_COACH_MODEL=tiny-aya-global
ECHOCOACH_TTS_PRESET=piper-multilingual
ECHOCOACH_REALTIME_TTS_PRESET=vibevoice-realtime-0.5b
```

| Mode | Purpose |
| ---- | ------- |
| **Explain** | Tutor any topic in plain language |
| **Lesson coach** | Discuss and outline lesson content |

Turn-based (not full duplex): speak → wait → hear reply. **Auto-speak replies** synthesizes TTS each turn when the language has a Piper voice.

Pitch metrics and monologue analysis live in **Classic UI → EchoCoach** (`/classic`).

### TeacherVoice — Classic UI (turn-based)

The **TeacherVoice** tab in `/classic` is the legacy multi-turn voice teacher — same pipeline as Language lessons, plus **Pitch practice** mode.

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

## Gradio SDK local smoke test (matches HF Space build)

Before pushing to Hugging Face, verify the Gradio SDK entry point:

```bash
python -m venv .venv-gradio && source .venv-gradio/bin/activate
pip install -r requirements.txt
ACTIVE_MODEL=minicpm5-1b ALLOW_MODEL_SWITCH=false python app.py
```

Open [http://localhost:7860](http://localhost:7860) — Studio at `/`, Classic at `/classic`.

Day-to-day development can still use `uv run` (see above); this path mirrors what HF installs from `requirements.txt`.

---

## Hugging Face Space deployment (Gradio SDK + ZeroGPU)

The Space card metadata lives in the YAML frontmatter at the top of [README.md](README.md) (`sdk: gradio`, `app_file: app.py`).

### 1. Push code to GitHub

Make sure `main` contains at minimum:

- `app.py`, `requirements.txt`, `packages.txt`
- `README.md` (with `sdk: gradio`, `sdk_version`, `app_file: app.py`)
- `models.yaml`, `skills/`
- `apps/gradio-space/` and all `libs/*` packages

The root `Dockerfile` stays in the repo for a later Docker SDK deploy (see below).

### 2. Create the Space

1. Go to [build-small-hackathon](https://huggingface.co/build-small-hackathon)
2. **New Space**
3. Name: e.g. `lesson-agent` or `small-model-hackathon`
4. SDK: **Gradio** (Blank template)
5. Hardware: **ZeroGPU** (creator needs PRO/Team) or **GPU basic**
6. Link your GitHub repo, or push directly to the Space git remote

CLI alternative (if you have `hf` installed and org access):

```bash
hf repo create build-small-hackathon/<your-space-name> \
  --repo-type space \
  --space_sdk gradio
```

### 3. Set Space environment variables

In the Space **Settings → Variables and secrets**:

| Variable | Value |
| -------- | ----- |
| `ACTIVE_MODEL` | `minicpm5-1b` |
| `ALLOW_MODEL_SWITCH` | `false` |
| `RESEARCHMIND_DATA_DIR` | `/tmp/researchmind` |

Default preset in [`models.yaml`](models.yaml) is `minicpm5-1b` (transformers) — suitable for ZeroGPU.

### 4. Build and verify

HF installs from `requirements.txt` and runs root `app.py`. Check the **Logs** tab for:

- Successful pip install (first build may take several minutes — `llama-cpp-python` compiles)
- `Running on local URL: 0.0.0.0:7860`

Smoke test on the live Space:

1. **`/`** — Studio UI loads
2. **`/classic`** — all tabs render
3. Generate slides with a simple topic (e.g. "Photosynthesis, grade 8, 5 slides")
4. First LLM request may be slow (model download + ZeroGPU queue)

### 5. ZeroGPU notes

LLM handlers use `@spaces.GPU` via [`gradio_space/spaces_runtime.py`](apps/gradio-space/src/gradio_space/spaces_runtime.py). If you see **No CUDA GPUs are available**, an inference path is running outside a decorated handler.

Startup model preload is skipped on HF Gradio runtime; the first user request loads the model inside a GPU task.

### 6. Optional: persistent model cache

Attach a **Storage Bucket** in Space settings so Hub model weights survive restarts.

---

## Docker SDK deployment (later)

Both deploy paths live on the same branch. HF reads **one** `sdk:` from README — switch to Docker when you are ready for a dedicated-GPU Space.

1. Change [README.md](README.md) frontmatter to `sdk: docker`, `app_port: 7860` (remove `sdk_version` / `app_file`)
2. Create or reconfigure a Space with **Docker** SDK and **GPU basic** hardware
3. Set the same env vars (`ACTIVE_MODEL=minicpm5-1b`, etc.)

### Local Docker smoke test

```bash
docker build -t hackathon-space .
docker run --rm -p 7860:7860 \
  -e ACTIVE_MODEL=minicpm5-1b \
  -e ALLOW_MODEL_SWITCH=false \
  -e RESEARCHMIND_DATA_DIR=/tmp/researchmind \
  hackathon-space
```

Open [http://localhost:7860](http://localhost:7860) — Studio at `/`, Classic tabs at `/classic`. Stop with `Ctrl+C`.

To use a pre-downloaded local GGUF model inside Docker, mount it and set `MODEL_PATH`:

```bash
docker run --rm -p 7860:7860 \
  -v "$(pwd)/models:/app/models:ro" \
  -e MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  hackathon-space
```

---

## Troubleshooting


| Symptom                                  | Likely cause                      | Fix                                                                  |
| ---------------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| First chat hangs / slow                  | Model downloading from Hub        | Wait on Space; use Storage Bucket for cache                            |
| `Failed to load model` in chat           | Wrong `ACTIVE_MODEL` preset       | Use `minicpm5-1b` or valid key from `models.yaml`                    |
| Space build fails on pip install         | `llama-cpp-python` compile        | Check Logs; default preset avoids GGUF at runtime                    |
| Space build fails                        | Malformed README YAML             | Ensure `sdk: gradio` and `app_file: app.py` in README frontmatter    |
| No CUDA GPUs on ZeroGPU                  | Handler outside `@spaces.GPU`     | LLM entry points must use `gpu_task` in `spaces_runtime.py`          |
| Docker build fails on `llama-cpp-python` | Missing build tools               | Dockerfile installs `build-essential` and `cmake`                    |
| Port already in use locally              | Another process on 7860           | `PORT=7861 python app.py` or `uv run ...`                            |


---

## Entrypoint summary

| Environment | How to run |
| ----------- | ---------- |
| Local dev (uv) | `uv run --package gradio-space python -m gradio_space.app` |
| Local Gradio SDK smoke | `pip install -r requirements.txt && python app.py` |
| HF Gradio Space | HF runs root `app.py` automatically |
| Docker (later) | `docker run -p 7860:7860 hackathon-space` (after README `sdk: docker`) |


