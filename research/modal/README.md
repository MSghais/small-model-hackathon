# Modal finetune + benchmark

GPU fine-tuning and lm-eval on Modal for `openbmb/MiniCPM5-1B`, wrapping existing [`research/finetune.py`](../finetune.py) and `slm-lm-eval`.

Targets **Modal partner track** (cloud GPU jobs) and **Well-Tuned** (before/after benchmark on lesson data).

## One-time setup

```bash
pip install modal
modal setup
modal secret create huggingface HF_TOKEN=<your-hf-token>
```

Validate lockfile locally:

```bash
uv sync --group finetune --group lm-eval --package slm-evals
```

## Run the sweep

From **repo root**:

```bash
# Full sweep: baseline lm-eval → 3 dataset jobs → post-train lm-eval
modal run research/modal/finetune_app.py

# Smoke one job (20 steps)
modal run research/modal/finetune_app.py --job lesson-lora --max-steps 20

# Re-run lm-eval only (checkpoints already on Volume)
modal run research/modal/finetune_app.py --eval-only --job lesson-lora

# Parallel training (3 GPUs at once — higher cost)
modal run research/modal/finetune_app.py --parallel
```

Jobs are defined in [`experiments.yaml`](experiments.yaml) (`lesson-lora`, `alpaca-lora`, `smoltalk-lora`).

## Pull results locally

```bash
modal volume get slm-finetune lesson-lora ./models/finetuned/lesson-lora
modal volume get slm-finetune results/lm_eval ./results/lm_eval
```

Wire the Space:

```bash
# models.yaml already has minicpm5-1b-lesson-lora; copy adapter to expected path:
cp -r ./models/finetuned/lesson-lora ./models/finetuned/minicpm5-1b-lora
# ACTIVE_MODEL=minicpm5-1b-lesson-lora
```

## Modal GPU Notebook

For interactive exploration, open [`research/notebook/minicpm5-modal-finetune.ipynb`](../notebook/minicpm5-modal-finetune.ipynb) in a [Modal GPU Notebook](https://modal.com/docs/guide/notebooks-modal):

1. Clone this repo into the notebook environment.
2. Run `uv sync --group finetune --group lm-eval --package slm-evals`.
3. Smoke-train with `research/finetune.py --preset minicpm5-1b --mode lora --max_steps 20`.

## Architecture

| Modal resource | Mount / role |
| -------------- | ------------ |
| Volume `hf-cache` | `/root/.cache/huggingface` — model + dataset cache |
| Volume `slm-finetune` | `/vol/finetuned` — adapters, `training_results.json`, lm-eval output |
| Secret `huggingface` | `HF_TOKEN` for Hub downloads |
| GPU `A10G` | Default for train + eval functions |

## Hackathon checklist

1. Screenshot or link to Modal app run (`slm-finetune-benchmark`).
2. `comparison.md` from `results/lm_eval/*__modal-lm-eval/` showing base vs lesson-LoRA.
3. Optional: demo video from Modal Notebook training cell.
