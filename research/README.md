# Research

Experimental code for **fine-tuning**, **ensemble architectures**, and **agentic benchmarks**. Nothing here is wired into the Gradio Lesson Agent by default — use it to train models, probe JEPA/world-model ideas, and score checkpoints against public benchmarks.

| Path | Purpose |
| ---- | ------- |
| [`finetune.py`](finetune.py) | LoRA / QLoRA / full fine-tune on chat or instruction data |
| [`ensemble/`](ensemble/) | JEPA + world-model ensemble experiments (uv package `ensemble`) |
| [`evals/`](evals/) | SLM agentic benchmark suite — BFCL, τ-bench, GAIA, SWE-bench (uv package `slm-evals`) |
| [`data/`](data/) | Shared JSONL datasets for finetune and ensemble harnesses |

## Quick links

- **[USAGE.md](USAGE.md)** — install groups, commands, and typical workflows
- **[docs/overview.md](docs/overview.md)** — how the pieces fit together
- **[ensemble/README.md](ensemble/README.md)** — ensemble smoke tests and harnesses
- **[evals/USAGE.md](evals/USAGE.md)** — benchmark CLI, configs, and results
- **[evals/docs/benchmarks.md](evals/docs/benchmarks.md)** — what each benchmark measures

## Install (from repo root)

```bash
# Everything you need for research scripts
uv sync --group finetune --group ensemble --group evals
```

Individual groups:

| Group | Command | Enables |
| ----- | ------- | ------- |
| `finetune` | `uv sync --group finetune` | `research/finetune.py` (LoRA, QLoRA, merge) |
| `ensemble` | `uv sync --group ensemble` | `research/ensemble/` package |
| `evals` | `uv sync --group evals` | `research/evals/` package (`slm-benchmark`) |

## Typical workflow

```text
research/data/education-lesson-chat.jsonl
        │
        ▼
  research/finetune.py  ──►  models/finetuned/<preset>-lora/
        │
        ├──► research/evals/  (BFCL, τ-bench, GAIA, SWE-bench)
        │
        └──► research/ensemble/  (JEPA / world-model ablations)
```

See [USAGE.md](USAGE.md) for copy-paste commands.
