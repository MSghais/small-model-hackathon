---
title: Lesson Agent
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.16.0"
app_file: app.py
python_version: "3.12"
pinned: false
license: apache-2.0
tags:
  - build-small-hackathon
  - backyard-ai
  - tiny-titan
  - best-agent
  - best-demo
  - openbmb
  - sharing-is-caring
  - off-the-grid
  - off-brand
---

# Lesson Agent

**Backyard AI** Gradio Space for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).

A local skill-based agent helps a teacher you know turn a **topic + grade level** into a downloadable **PowerPoint** — powered by a small transformers model (`MiniCPM5-1B` by default), no cloud LLM API.

See **[USAGE.md](USAGE.md)** for local run, Gradio SDK / ZeroGPU Space deployment, and Docker (later).

**Demo video:** [https://www.youtube.com/watch?v=bwtOiZvJ-7k](https://www.youtube.com/watch?v=bwtOiZvJ-7k)

**Blog post:** [Small Models, Bounded Jobs](https://huggingface.co/blog/build-small-hackathon/lessonagent-opennotebook) — Hugging Face Build Small Hackathon write-up

**X post:** [https://x.com/MSG_Encrypted/status/2066570320861921748](https://x.com/MSG_Encrypted/status/2066570320861921748)


**Github:** [https://github.com/MSghais/small-model-hackathon/](https://github.com/MSghais/small-model-hackathon/)

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Python 3.12

## Quick start

```bash
uv sync --all-packages
cp .env.example .env   # optional: edit model settings

# Run Gradio locally
uv run --package gradio-space python -m gradio_space.app
```

Open [http://localhost:7860](http://localhost:7860).

- **Lesson slides** — topic, grade, slide count → downloadable PowerPoint
- **Research Agent** — scrape/index sources into MemRAG, then ask questions offline with citations

### Studio UI (Off Brand track)

The default landing page is a **custom AI Studio workspace** at `/` — not default Gradio chrome. It uses **Gradio 6 Server mode** (`gradio.Server`): Material 3 layout, sidebar + workspace (Research → Slides → Language lessons), and `@server.api` endpoints wired to the same Python backends as Classic.

- **`/`** — Studio UI (ingest sources, generate slides, **Language lessons** multilingual coach)
- **`/classic`** — full Gradio Blocks app (TeacherVoice, EchoCoach pitch analysis, settings, Chat debug)

See [apps/gradio-space/README.md](apps/gradio-space/README.md) for API names and a 2-minute judge demo script.

### Modal + Fine-tuning track (Well-Tuned)

Cloud GPU **train → eval → gate → publish** for a skill-matrix of QLoRA adapters on `openbmb/MiniCPM5-1B` — no local CUDA required. Each job in [`research/modal/experiments.yaml`](research/modal/experiments.yaml) (math, science, coding, reasoning, teaching, …) fine-tunes with [`research/finetune.py`](research/finetune.py), benchmarks with `slm-lm-eval`, gates on per-skill `goals`, and publishes passing adapters to the Hub.

- **Modal (partner track)** — `modal run` / warm GPU worker, Volume artifacts, optional [Modal Notebook](research/notebook/minicpm5-modal-finetune.ipynb)
- **Well-Tuned badge** — before/after lm-eval per skill + gated Hub publish (`MSGEncrypted/minicpm5-1b-<skill>-lora`)

Full runbook: [`research/modal/README.md`](research/modal/README.md) · agent loop: [`research/modal/SERVER.md`](research/modal/SERVER.md) · local research overview: [`research/USAGE.md`](research/USAGE.md)

```bash
uv sync --group modal
modal setup && modal secret create huggingface HF_TOKEN=<token>

modal run research/modal/server_app.py --ping                       # health check
modal run research/modal/server_app.py --job math-lora --max-steps 20 --no-publish   # cheap smoke
modal run research/modal/server_app.py --pipeline                   # full sweep: baselines → train → eval → gate → publish
```

Pull a passing adapter into the Space: `modal volume get slm-finetune math-lora ./models/finetuned/minicpm5-1b-lora`, then set `ACTIVE_MODEL=minicpm5-1b-lesson-lora`.

### Llama track (Llama Champion + Off-the-Grid)

The same OpenBMB **MiniCPM-V 4.6** model runs on **llama.cpp** via the [`minicpm-v-4.6-gguf`](models.yaml) preset — GGUF weights from [`openbmb/MiniCPM-V-4.6-gguf`](https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf) (~529 MB Q4_K_M). No cloud LLM API; inference stays fully local through [`libs/inference/src/inference/llama_cpp.py`](libs/inference/src/inference/llama_cpp.py).

| Preset | Backend | Use case |
| ------ | ------- | -------- |
| `minicpm-v-4.6` | transformers | Full VLM (image/video) via Hugging Face |
| `minicpm-v-4.6-gguf` | llama.cpp | **Llama Champion** badge; text on all tabs today |

**Space (judges):** pin the GGUF preset — no runtime switching for visitors.

```bash
ACTIVE_MODEL=minicpm-v-4.6-gguf
ALLOW_MODEL_SWITCH=false
```

**Local dev:** switch backends at runtime without restarting.

```bash
ALLOW_MODEL_SWITCH=true
ACTIVE_MODEL=minicpm-v-4.6          # transformers startup default
# Settings or Chat → select minicpm-v-4.6-gguf for llama.cpp
```

Prefetch weights (optional):

```bash
uv run python scripts/download_model.py --preset minicpm-v-4.6-gguf
```

See [USAGE.md](USAGE.md) (section *Switching models locally*) for Classic and Studio UI details.

## How it works

1. **Skill** — `skills/education-pptx/SKILL.md` (Hermes / agentskills.io format)
2. **LLM** — local model drafts a JSON slide outline
3. **Tool** — `create_pptx` builds the file with `python-pptx`
4. **Trace** — JSON log saved under `outputs/traces/` for the Sharing is Caring badge

```text
apps/gradio-space/   # Gradio tabs (Lesson slides, Research Agent, Chat debug)
libs/agent/          # Skill agent runner, tools, trace recorder
libs/researchmind/   # Scraper, chunk/embed, MemRAG SQLite store, retrieval
libs/inference/      # Transformers + llama.cpp backends
skills/              # SKILL.md + references/ + scripts/ per task
research/            # Fine-tune and agentic evals (optional)
```

### ResearchMind (offline after ingest)

1. **Skills** — `skills/scrape-web`, `scrape-pdf`, `extract-content`, `research-mind`
2. **Ingest** — URL/PDF/DOCX or topic → (optional LLM URL suggest + confirm, or auto search) → chunk + embed → SQLite
3. **Q&A** — local model + retrieved chunks with `[n]` citations (no network at chat time)
4. **Memory** — persists under `RESEARCHMIND_DATA_DIR` (default `outputs/researchmind`)

Optional research tooling (not required for the Space): see [research/USAGE.md](research/USAGE.md).

## Environment variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `ACTIVE_MODEL` | `minicpm5-1b` | Preset key from `models.yaml` (use `minicpm-v-4.6-gguf` for Llama track) |
| `ALLOW_MODEL_SWITCH` | `false` | Set `true` locally to switch presets in Settings / Chat |
| `AGENT_OUTPUTS_DIR` | `/tmp/agent_outputs` | Generated `.pptx` files |
| `AGENT_TRACES_DIR` | `outputs/traces` | Agent trace JSON |
| `SKILLS_DIR` | `./skills` | Skill definitions root |
| `RESEARCHMIND_DATA_DIR` | `outputs/researchmind` | MemRAG DB and raw snapshots |
| `RESEARCHMIND_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `RESEARCHMIND_AUTO_SEARCH` | `false` | Default auto DuckDuckGo ingest |

See [`.env.example`](.env.example) and [`models.yaml`](models.yaml) for model presets.

## Hugging Face Space deployment

1. Create a Space under [build-small-hackathon](https://huggingface.co/build-small-hackathon) with **Gradio** SDK (Blank template).
2. Link this repository — HF builds from root `app.py` + `requirements.txt` (README YAML above).
3. Hardware: **ZeroGPU** for burst GPU inference, or **GPU basic** for always-on GPU.
4. Set `ACTIVE_MODEL=minicpm5-1b` (or `minicpm-v-4.6-gguf` for [Llama track](#llama-track-llama-champion--off-the-grid)), `ALLOW_MODEL_SWITCH=false`, `RESEARCHMIND_DATA_DIR=/tmp/researchmind`.

A root `Dockerfile` is kept for a later **Docker SDK** deploy (flip README to `sdk: docker`). See [USAGE.md](USAGE.md).

## Hackathon tracks & checklist

| Track | What we ship |
| ----- | ------------ |
| **Backyard AI** (primary) | Lesson slide builder for a teacher you know — topic + grade → downloadable `.pptx` |
| **Off Brand** | Custom Studio UI at `/` (Gradio 6 Server mode, not default Gradio chrome) |
| **Modal** (partner) | GPU `train → eval → gate → publish` on [Modal](https://modal.com) — [`research/modal/`](research/modal/) |
| **Well-Tuned** (finetuning) | Skill-matrix QLoRA adapters on MiniCPM5-1B, lm-eval gates, Hub publish |
| **Llama Champion** | `minicpm-v-4.6-gguf` on llama.cpp — same OpenBMB VLM family, local GGUF inference |

- Space live under build-small-hackathon
- Demo video: [YouTube](https://www.youtube.com/watch?v=bwtOiZvJ-7k) — real user enters topic → download `.pptx` → show agent trace
- Blog post: [Small Models, Bounded Jobs](https://huggingface.co/blog/build-small-hackathon/lessonagent-opennotebook)
- Social post published: [X](https://x.com/MSG_Encrypted/status/2066570320861921748)
- Submission by **June 15, 2026**

### Badge targets

- **Best Agent** — skill loop + `create_pptx` tool
- **Tiny Titan** — MiniCPM5 1B (≤4B)
- **OpenBMB** — `openbmb/MiniCPM5-1B`
- **Sharing is Caring** — upload traces with `scripts/upload_trace.py`
- **Off-the-Grid** — local inference only (no cloud LLM API)
- **Llama Champion** — llama.cpp backend with [`openbmb/MiniCPM-V-4.6-gguf`](https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf); see [Llama track](#llama-track-llama-champion--off-the-grid)
- **Well-Tuned** — per-skill QLoRA adapters trained + gated + published via the [Modal + Fine-tuning track](#modal--fine-tuning-track-well-tuned)
- **Modal** — same pipeline; see [`research/modal/README.md`](research/modal/README.md)

## Agent trace upload

```bash
uv run python scripts/upload_trace.py --repo-id YOUR_USER/build-small-agent-traces
```

## Demo video script

1. Introduce the teacher and the problem (building a 5-slide lesson takes 30+ minutes).
2. Open **Lesson slides**, enter topic + grade, click **Generate**.
3. Show outline preview and download the `.pptx`.
4. Expand the agent trace JSON — local model, no cloud API.
