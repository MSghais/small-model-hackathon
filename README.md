---

## title: Lesson Agent
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0

# Lesson Agent

**Backyard AI** Gradio Space for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).

A local skill-based agent helps a teacher you know turn a **topic + grade level** into a downloadable **PowerPoint** — powered by a small transformers model (`MiniCPM5-1B` by default), no cloud LLM API.

See **[USAGE.md](USAGE.md)** for local run, Docker smoke test, and HF Space deployment.

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

Open [http://localhost:7860](http://localhost:7860). Use the **Lesson slides** tab: enter a topic, grade, and slide count. The model loads on first generate.

## How it works

1. **Skill** — `skills/education-pptx/SKILL.md` (Hermes / agentskills.io format)
2. **LLM** — local model drafts a JSON slide outline
3. **Tool** — `create_pptx` builds the file with `python-pptx`
4. **Trace** — JSON log saved under `outputs/traces/` for the Sharing is Caring badge

```text
apps/gradio-space/   # Gradio tabs (Lesson slides + Chat debug)
libs/agent/          # Skill agent runner, tools, trace recorder
libs/inference/      # Transformers + llama.cpp backends
skills/              # SKILL.md task definitions
research/            # Fine-tune, ensemble experiments, agentic evals (optional)
```

Optional research tooling (not required for the Space): see [research/USAGE.md](research/USAGE.md).

## Environment variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `ACTIVE_MODEL` | `minicpm5-1b` | Preset key from `models.yaml` |
| `AGENT_OUTPUTS_DIR` | `/tmp/agent_outputs` | Generated `.pptx` files |
| `AGENT_TRACES_DIR` | `outputs/traces` | Agent trace JSON |
| `SKILLS_DIR` | `./skills` | Skill definitions root |

See [`.env.example`](.env.example) and [`models.yaml`](models.yaml) for model presets.

## Hugging Face Space deployment

1. Create a Space under [build-small-hackathon](https://huggingface.co/build-small-hackathon) with **Docker** SDK.
2. Link this repository (root `Dockerfile` + root `README.md` YAML above).
3. Hardware: **GPU basic** recommended for transformers (`minicpm5-1b`).
4. Optional secrets: `ACTIVE_MODEL`, `N_GPU_LAYERS` (if using GGUF preset).

```bash
docker build -t hackathon-space .
docker run --rm -p 7860:7860 -e ACTIVE_MODEL=minicpm5-1b hackathon-space
```

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
