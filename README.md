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
---

# Lesson Agent

**Backyard AI** Gradio Space for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).

A local skill-based agent helps a teacher you know turn a **topic + grade level** into a downloadable **PowerPoint** — powered by a small transformers model (`MiniCPM5-1B` by default), no cloud LLM API.

See **[USAGE.md](USAGE.md)** for local run, Gradio SDK / ZeroGPU Space deployment, and Docker (later).

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

### Studio UI (Off Brand track)

The default landing page is a **custom AI Studio workspace** at `/` — not default Gradio chrome. It uses **Gradio 6 Server mode** (`gradio.Server`): Material 3 layout, sidebar + three-column workspace (Research → Slides → Voice/Coach), and `@server.api` endpoints wired to the same Python backends as Classic.

- **`/`** — Studio UI (ingest sources, generate slides, TeacherVoice, EchoCoach)
- **`/classic`** — full Gradio Blocks app (all tabs, settings, Chat debug)

See [apps/gradio-space/README.md](apps/gradio-space/README.md) for API names and a 2-minute judge demo script.

- **Lesson slides** — topic, grade, slide count → downloadable PowerPoint
- **Research Agent** — scrape/index sources into MemRAG, then ask questions offline with citations

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
research/            # Fine-tune, ensemble experiments, agentic evals (optional)
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
| `ACTIVE_MODEL` | `minicpm5-1b` | Preset key from `models.yaml` |
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
4. Set `ACTIVE_MODEL=minicpm5-1b`, `ALLOW_MODEL_SWITCH=false`, `RESEARCHMIND_DATA_DIR=/tmp/researchmind`.

A root `Dockerfile` is kept for a later **Docker SDK** deploy (flip README to `sdk: docker`). See [USAGE.md](USAGE.md).

## Hackathon checklist

- **Track:** Backyard AI — lesson slide builder for a teacher you know
- Space live under build-small-hackathon
- Demo video: real user enters topic → download `.pptx` → show agent trace
- Social post published
- Submission by **June 15, 2026**

### Badge targets

- **Best Agent** — skill loop + `create_pptx` tool
- **Tiny Titan** — MiniCPM5 1B (≤4B)
- **OpenBMB** — `openbmb/MiniCPM5-1B`
- **Sharing is Caring** — upload traces with `scripts/upload_trace.py`
- **Off-the-Grid** — local inference only (no cloud LLM API)
- **Well-Tuned** — optional fine-tuned preset in `models.yaml` (Phase 2)

## Agent trace upload

```bash
uv run python scripts/upload_trace.py --repo-id YOUR_USER/build-small-agent-traces
```

## Demo video script

1. Introduce the teacher and the problem (building a 5-slide lesson takes 30+ minutes).
2. Open **Lesson slides**, enter topic + grade, click **Generate**.
3. Show outline preview and download the `.pptx`.
4. Expand the agent trace JSON — local model, no cloud API.
