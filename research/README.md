# Research

Experimental code for **fine-tuning** and **agentic benchmarks**. Nothing here is wired into the Gradio Lesson Agent by default — use it to train models and score checkpoints against public benchmarks.

| Path | Purpose |
| ---- | ------- |
| [`finetune.py`](finetune.py) | LoRA / QLoRA / full fine-tune on chat or instruction data |
| [`evals/`](evals/) | SLM agentic benchmark suite — BFCL, τ-bench, GAIA, SWE-bench (uv package `slm-evals`) |
| [`data/`](data/) | Shared JSONL datasets for finetune and evals |

## Quick links

- **[USAGE.md](USAGE.md)** — install groups, commands, and typical workflows
- **[docs/overview.md](docs/overview.md)** — how the pieces fit together
- **[evals/USAGE.md](evals/USAGE.md)** — benchmark CLI, configs, and results
- **[evals/docs/benchmarks.md](evals/docs/benchmarks.md)** — what each benchmark measures

## Install (from repo root)

```bash
# All research tooling
uv sync --group finetune --group evals --group lm-eval
```

Individual groups:

| Group | Command | Enables |
| ----- | ------- | ------- |
| `finetune` | `uv sync --group finetune` | `research/finetune.py` (LoRA, QLoRA, merge) |
| `evals` | `uv sync --group evals` | `research/evals/` package (`slm-benchmark`) |
| `lm-eval` | `uv sync --group lm-eval` | `slm-lm-eval` CLI (GSM8K, ARC, HellaSwag, …) |

## Typical workflow

```text
research/data/education-lesson-chat.jsonl
        │
        ▼
  research/finetune.py  ──►  models/finetuned/<preset>-lora/
        │
        └──► research/evals/  (BFCL, τ-bench, GAIA, SWE-bench, lm-eval)
```

See [USAGE.md](USAGE.md) for copy-paste commands.
