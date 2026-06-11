# Research usage

How to run fine-tuning, ensemble experiments, and agentic benchmarks under `research/`. All commands assume the **repo root** as the working directory unless noted.

The Lesson Agent app lives in `apps/gradio-space/` — see root [USAGE.md](../USAGE.md). Research code is optional and isolated here.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.12
- GPU recommended for real-model runs (CPU works for smoke tests and `tiny` backends)
- Hugging Face Hub access for model downloads and some benchmark datasets

## Install dependency groups

```bash
# All research tooling
uv sync --group finetune --group ensemble --group evals

# Or one at a time
uv sync --group finetune
uv sync --group ensemble
uv sync --group evals
```

| Group | Package / script | What it adds |
| ----- | ---------------- | ------------ |
| `finetune` | `research/finetune.py` | `peft`, `datasets`, `bitsandbytes` (QLoRA) |
| `ensemble` | `ensemble` workspace member | JEPA / world-model ensemble + harnesses |
| `evals` | `slm-evals` workspace member | `slm-benchmark` CLI |

---

## 1. Fine-tuning (`research/finetune.py`)

Single script for **full**, **LoRA**, and **QLoRA** training. Defaults to the lesson-agent chat dataset at `research/data/education-lesson-chat.jsonl` and writes checkpoints under `models/finetuned/`.

### Model resolution (first match wins)

1. `--model <hf-id-or-path>`
2. `--preset <key>` from root `models.yaml`
3. Env: `FINETUNE_MODEL`, `MODEL_ID`, or `BASE`
4. `ACTIVE_MODEL` preset from `.env`

### Quick start

```bash
# LoRA on default lesson chat data, 1 epoch
uv run python research/finetune.py --preset minicpm5-1b --mode lora --epochs 1

# Smoke run (50 steps)
uv run python research/finetune.py --mode lora --max_steps 50

# QLoRA on a Hub instruction dataset
uv run python research/finetune.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset tatsu-lab/alpaca --format alpaca \
  --mode qlora --epochs 1

# Merge LoRA adapter into standalone weights
uv run python research/finetune.py \
  --merge ./models/finetuned/minicpm5-1b-lora \
  --out ./models/finetuned/minicpm5-1b-merged
```

### Dataset formats (`--format`)

| Format | Expected columns |
| ------ | ---------------- |
| `chat` | `messages`: `[{"role": "...", "content": "..."}]` |
| `alpaca` | `instruction`, optional `input`, `output` |
| `prompt` | `prompt` / `completion` (or `response`) |
| `text` | `text`, or a plain `.txt` file |

Local files: `.json`, `.jsonl`, `.csv`, `.txt`. Hub ids: any `datasets` repo id.

### Outputs

Training writes to `<out>/` (default `./models/finetuned/<preset>-<mode>/`):

- Adapter or full weights
- `training_results.json` — train/eval loss, perplexity, `result_score` (0–100)

### Env vars

| Variable | Description |
| -------- | ----------- |
| `FINETUNE_PRESET` | Preset key from `models.yaml` |
| `FINETUNE_DATASET` | Override dataset path or Hub id |
| `FINETUNE_DATASET_CONFIG` | Hub config name |
| `FINETUNE_DATASET_SPLIT` | Hub split (e.g. `train[:500]`) |
| `ACTIVE_MODEL` | Fallback preset when `--preset` omitted |

---

## 2. Ensemble experiments (`research/ensemble/`)

JEPA and world-model ensemble prototypes: small LLM + embedding memory + latent predictors + energy-based draft selection. **Not connected to the Gradio app.**

Install: `uv sync --group ensemble`

### Tier 1 — CPU smoke (no Hub download)

```bash
uv run --package ensemble python -m ensemble.jepa_ensemble tiny
uv run --package ensemble python -m ensemble.world_ensemble tiny
bash research/ensemble/scripts/smoke.sh
```

### Tier 2 — Real small model

```bash
uv run --package ensemble python -m ensemble.jepa_ensemble Qwen/Qwen2.5-0.5B-Instruct
uv run --package ensemble python -m ensemble.world_ensemble Qwen/Qwen2.5-0.5B-Instruct
```

### Tier 3 — Benchmark harnesses

Uses `research/data/benchmark-qa.jsonl` (questions) and `benchmark-kb.jsonl` (retrieval snippets).

```bash
# JEPA track — toy
uv run --package ensemble python -m ensemble.eval.jepa_harness \
  --llm tiny --toy --limit 20 --n_drafts 8

# JEPA track — education QA
uv run --package ensemble python -m ensemble.eval.jepa_harness \
  --llm Qwen/Qwen2.5-0.5B-Instruct \
  --qa research/data/benchmark-qa.jsonl \
  --kb research/data/benchmark-kb.jsonl \
  --limit 50 --n_drafts 8

# World-model track
uv run --package ensemble python -m ensemble.eval.world_harness \
  --llm tiny --toy --limit 20 --n_drafts 8
```

More detail: [ensemble/README.md](ensemble/README.md), [docs/overview.md](docs/overview.md).

### Legacy shims

Top-level files re-export the package for old scripts:

- `research/llm_emb_jepa_ensemble_pluggable.py` → `ensemble.jepa_ensemble`
- `research/world_model_ensemble.py` → `ensemble.world_ensemble`
- `research/eval_harness.py` → `ensemble.eval.jepa_harness`

Prefer `uv run --package ensemble python -m ensemble.<module>`.

---

## 3. Agentic benchmarks (`research/evals/`)

Evaluate local HuggingFace checkpoints on BFCL, τ-bench, GAIA, and SWE-bench Verified.

Install: `uv sync --group evals`

```bash
# Smoke test (20 samples, two benchmarks)
uv run --package slm-evals slm-benchmark \
  --model openbmb/MiniCPM5-1B \
  --benchmarks bfcl tau_bench \
  --max-samples 20

# Full config-driven run
uv run --package slm-evals slm-benchmark \
  --config research/evals/configs/experiment_001.yaml
```

Full reference: [evals/USAGE.md](evals/USAGE.md).

---

## Shared data (`research/data/`)

| File | Used by | Format |
| ---- | ------- | ------ |
| `education-lesson-chat.jsonl` | `finetune.py` default | Chat messages for lesson agent |
| `benchmark-qa.jsonl` | Ensemble harnesses | `question`, `answer`, `domain` |
| `benchmark-kb.jsonl` | Ensemble harnesses | Retrieval snippets for memory routing |

---

## Suggested end-to-end pipeline

1. **Baseline eval** — score the base preset before training:
   ```bash
   uv run --package slm-evals slm-benchmark \
     --model openbmb/MiniCPM5-1B --benchmarks bfcl --max-samples 50
   ```

2. **Fine-tune** on lesson data:
   ```bash
   uv run python research/finetune.py --preset minicpm5-1b --mode lora --epochs 1
   ```

3. **Re-eval** the merged or adapter-backed checkpoint:
   ```bash
   uv run --package slm-evals slm-benchmark \
     --model ./models/finetuned/minicpm5-1b-lora \
     --benchmarks bfcl tau_bench --max-samples 50
   ```

4. **Optional** — probe ensemble ideas on the same QA/KB files:
   ```bash
   bash research/ensemble/scripts/smoke.sh
   ```

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `No module named 'ensemble'` | `uv sync --group ensemble` |
| `slm-benchmark: command not found` | `uv sync --group evals` |
| CUDA OOM during finetune | Use `--mode qlora` or reduce batch size in script args |
| BFCL / GAIA download slow | Set `max_samples` low first; cache HF datasets under `~/.cache/huggingface` |
| SWE-bench Docker errors | Keep `full_eval: false` in YAML unless `swebench` + Docker are installed |
| τ-bench API costs | Keep `use_llm_user: false` (rule-based user simulator) |
