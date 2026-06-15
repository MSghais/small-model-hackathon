# Research usage

How to run fine-tuning and agentic benchmarks under `research/`. All commands assume the **repo root** as the working directory unless noted.

The Lesson Agent app lives in `apps/gradio-space/` — see root [USAGE.md](../USAGE.md). Research code is optional and isolated here.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.12
- GPU recommended for real-model runs (CPU works for smoke tests)
- Hugging Face Hub access for model downloads and some benchmark datasets

## Install dependency groups

```bash
# All research tooling
uv sync --group finetune --group evals --group lm-eval

# Or one at a time
uv sync --group finetune
uv sync --group evals
uv sync --group lm-eval
```

| Group | Package / script | What it adds |
| ----- | ---------------- | ------------ |
| `finetune` | `research/finetune.py` | `peft`, `datasets`, `bitsandbytes` (QLoRA) |
| `evals` | `slm-evals` workspace member | `slm-benchmark` CLI |
| `lm-eval` | `slm-evals[lm-eval]` | `slm-lm-eval` CLI (GSM8K, ARC, HellaSwag, …) |
| `modal` | `research/modal/finetune_app.py` | Cloud GPU train + eval via [Modal](https://modal.com/docs/guide) |

---

## 0. Modal cloud GPU (`research/modal/`)

Run fine-tuning and lm-eval **without local CUDA**. Wraps the same `finetune.py` and `slm-lm-eval` scripts; saves LoRA adapters to Modal Volume `slm-finetune`.

```bash
uv sync --group modal
modal setup
modal secret create huggingface HF_TOKEN=<token>

# Smoke train on Modal
modal run research/modal/finetune_app.py --job lesson-lora --max-steps 20

# Download adapter to repo path expected by models.yaml
modal volume get slm-finetune lesson-lora ./models/finetuned/minicpm5-1b-lora

# Publish to Hugging Face Hub
huggingface-cli upload your-user/minicpm5-1b-lesson-lora \
  ./models/finetuned/minicpm5-1b-lora . --repo-type model
```

Full guide (Volume layout, merge, Space deploy): **[modal/README.md](modal/README.md)**.

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

## 2. Agentic benchmarks (`research/evals/`)

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

## 3. Academic benchmarks (`slm-lm-eval`)

Standard lm-evaluation-harness tasks (ARC, HellaSwag, GSM8K, …) for base presets, LoRA adapters, and merged checkpoints.

Install: `uv sync --group lm-eval`

Profile guide: [evals/docs/eval_profiles.md](evals/docs/eval_profiles.md)

```bash
# List claim-matched profiles (reasoning, code, understanding, …)
uv run --package slm-evals slm-lm-eval --list-profiles

# Run by profile name
uv run --package slm-evals slm-lm-eval \
  --profile reasoning \
  --preset minicpm5-1b \
  --experiment-name minicpm5-1b__reasoning-baseline

# Smoke (25 samples, arc_easy + hellaswag)
uv run --package slm-evals slm-lm-eval \
  --profile smoke \
  --preset minicpm5-1b \
  --experiment-name minicpm5-1b__smoke

# Full profile
uv run --package slm-evals slm-lm-eval \
  --config research/evals/configs/lm_eval_minicpm5.yaml \
  --preset minicpm5-1b-lesson-lora \
  --experiment-name minicpm5-1b-lora__v1 \
  --compare-to results/lm_eval/minicpm5-1b__baseline/results.json
```

Post-training hook:

```bash
uv run python research/finetune.py \
  --preset minicpm5-1b --mode lora --max_steps 50 \
  --lm-eval-after \
  --lm-eval-baseline minicpm5-1b
```

Full reference: [evals/USAGE.md](evals/USAGE.md#lm-evaluation-harness-slm-lm-eval).

---

## Shared data (`research/data/`)

| File | Used by | Format |
| ---- | ------- | ------ |
| `education-lesson-chat.jsonl` | `finetune.py` default | Chat messages for lesson agent |
| `benchmark-qa.jsonl` | Optional domain QA evals | `question`, `answer`, `domain` |
| `benchmark-kb.jsonl` | Optional retrieval snippets | KB entries for domain QA |

---

## Suggested end-to-end pipeline

1. **Baseline lm-eval** — academic benchmarks on the base preset (pinned seed):
   ```bash
   uv run --package slm-evals slm-lm-eval \
     --config research/evals/configs/lm_eval_compare_study.yaml \
     --preset minicpm5-1b \
     --experiment-name minicpm5-1b__baseline
   ```

2. **Baseline agentic eval** (optional):
   ```bash
   uv run --package slm-evals slm-benchmark \
     --model openbmb/MiniCPM5-1B --benchmarks bfcl --max-samples 50
   ```

3. **Fine-tune** on lesson data:
   ```bash
   uv run python research/finetune.py --preset minicpm5-1b --mode lora --epochs 1
   ```

4. **Re-eval candidate** with the same lm-eval config:
   ```bash
   uv run --package slm-evals slm-lm-eval \
     --config research/evals/configs/lm_eval_compare_study.yaml \
     --preset minicpm5-1b-lesson-lora \
     --experiment-name minicpm5-1b-lora__v1 \
     --compare-to results/lm_eval/minicpm5-1b__baseline/results.json
   ```

### Verification checklist

- Use the **same** lm-eval YAML (`tasks`, `num_fewshot`, `limit`, `seed`) for baseline and candidate runs.
- Compare lm-eval `results.json` files with `--compare-to`; do not compare `training_results.json` `result_score` to lm-eval accuracy.
- For LoRA checkpoints, prefer `--preset minicpm5-1b-lesson-lora` (base + adapter) over passing the adapter dir alone to `--model`.
- Report mean ± std only after multiple training seeds; single-seed deltas are indicative, not conclusive.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `slm-benchmark: command not found` | `uv sync --group evals` |
| `slm-lm-eval: command not found` | `uv sync --group lm-eval` |
| CUDA OOM during finetune | Use `--mode qlora` or reduce batch size in script args |
| BFCL / GAIA download slow | Set `max_samples` low first; cache HF datasets under `~/.cache/huggingface` |
| SWE-bench Docker errors | Keep `full_eval: false` in YAML unless `swebench` + Docker are installed |
| τ-bench API costs | Keep `use_llm_user: false` (rule-based user simulator) |
