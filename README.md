---

## title: Small Model Hackathon
emoji: 🦙
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0

# Small Model Hackathon

Gradio chat Space for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon). Runs local inference with **llama.cpp** (GGUF) by default; optional **transformers** backend via env.

See **[USAGE.md](USAGE.md)** for local run, Docker smoke test, and HF Space deployment steps.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Python 3.12

## Quick start

```bash
uv sync --all-packages
cp .env.example .env   # optional: edit model settings

# Download GGUF for offline dev (optional)
uv run python scripts/download_model.py

# Run Gradio locally
uv run --package gradio-space python -m gradio_space.app
```

Open [http://localhost:7860](http://localhost:7860). The model downloads from Hugging Face Hub on the first chat message (or set `MODEL_PATH` to a local GGUF).

## Environment variables


| Variable            | Default                           | Description                                |
| ------------------- | --------------------------------- | ------------------------------------------ |
| `INFERENCE_BACKEND` | `llama_cpp`                       | `llama_cpp` or `transformers`              |
| `MODEL_REPO`        | `Qwen/Qwen2.5-3B-Instruct-GGUF`   | Hub repo for GGUF                          |
| `MODEL_FILE`        | `qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF filename                              |
| `MODEL_PATH`        | —                                 | Local GGUF path (skips Hub download)       |
| `N_CTX`             | `4096`                            | Context window                             |
| `N_GPU_LAYERS`      | `0`                               | GPU layers for llama.cpp (0 = CPU)         |
| `MODEL_ID`          | `Qwen/Qwen2.5-3B-Instruct`        | Used when `INFERENCE_BACKEND=transformers` |


See `[.env.example](.env.example)` for a full template.

## Monorepo layout

```text
apps/gradio-space/   # Gradio UI (HF Space entrypoint)
libs/inference/      # Swappable inference backends
scripts/             # Dev utilities
```

### Common commands

```bash
uv add --package gradio-space <package>
uv add --package inference <package>
uv run --package gradio-space python -m gradio_space.app
uv run python -c "from inference.factory import get_backend"
```

## Hugging Face Space deployment

1. Create a Space under [build-small-hackathon](https://huggingface.co/build-small-hackathon) with **Docker** SDK.
2. Link this repository (root `Dockerfile` + root `README.md` YAML above).
3. Hardware: start with **CPU basic**; upgrade to GPU if you set `N_GPU_LAYERS > 0`.
4. Add Space secrets: `MODEL_REPO`, `MODEL_FILE`, `N_CTX`, `N_GPU_LAYERS`.

```bash
# Optional local Docker smoke test
docker build -t hackathon-space .
docker run --rm -p 7860:7860 -e MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF hackathon-space
```

## Hackathon checklist

- Choose a track (Backyard AI or Thousand Token Wood)
- Space live under build-small-hackathon
- Demo video recorded
- Social post published
- Submission locked in by **June 15, 2026**

### Badge targets

- **Off-the-Grid** — local llama.cpp inference (default setup)
- **Llama Champion** — llama.cpp + GGUF model
- **Off-Brand** — custom UI via `gr.Server` (Phase 2)
- **Sharing is Caring** — agent traces dataset (Phase 2)

## Transformers backend (optional)

```bash
uv sync --package inference --extra transformers
INFERENCE_BACKEND=transformers MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
  uv run --package gradio-space python -m gradio_space.app
```

