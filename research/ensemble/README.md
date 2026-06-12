# Ensemble research package

JEPA and world-model ensemble experiments. Stays under `research/` — not wired into the Gradio agent.

See also: [../USAGE.md](../USAGE.md) · [../docs/overview.md](../docs/overview.md)

## Install

```bash
uv sync --group ensemble
```

## Tier 1 — Smoke (CPU, no HF download)

```bash
uv run --package ensemble python -m ensemble.jepa_ensemble tiny
uv run --package ensemble python -m ensemble.world_ensemble tiny
bash research/ensemble/scripts/smoke.sh
```

## Tier 2 — Micro demo (real small model)

```bash
uv run --package ensemble python -m ensemble.jepa_ensemble Qwen/Qwen2.5-0.5B-Instruct
uv run --package ensemble python -m ensemble.world_ensemble Qwen/Qwen2.5-0.5B-Instruct
```

## Pretrain + save (LLM + emb + JEPA)

Joint training writes a full checkpoint to `models/ensemble/<name>/`:

```bash
# CPU smoke (tiny backend, no HF download)
uv run --package ensemble ensemble-pretrain \
  --llm tiny --steps 50 --no-kb \
  --out models/ensemble/jepa-smoke

# Uses ACTIVE_MODEL / BASE / LLM_PATH from .env + models.yaml by default
uv run --package ensemble ensemble-pretrain \
  --data research/data/education-lesson-chat.jsonl \
  --kb research/data/benchmark-kb.jsonl \
  --steps 200

# Override base LLM explicitly
uv run --package ensemble ensemble-pretrain \
  --llm Qwen/Qwen2.5-0.5B-Instruct --steps 200
```

Checkpoint layout: `manifest.json`, `aux.pt` (emb/jepa/bridge/router), `llm/` (PEFT adapters).

Benchmark the saved ensemble with **slm-evals** (auto-detects `manifest.json`):

```bash
uv run --package slm-evals slm-benchmark \
  --model ./models/ensemble/jepa-lesson-pretrain \
  --model-type ensemble \
  --benchmarks bfcl tau_bench --max-samples 20

# Or use the template config
uv run --package slm-evals slm-benchmark \
  --config research/evals/configs/ensemble_jepa_lesson.yaml
```

Compare against a base HF model by running the same config with `model_type: hf` and `model_path: openbmb/MiniCPM5-1B`.

## Tier 3 — Benchmark

### JEPA ablation ladder

```bash
# Toy (no download)
uv run --package ensemble python -m ensemble.eval.jepa_harness \
  --llm tiny --toy --limit 20 --n_drafts 8

# Education QA set
uv run --package ensemble python -m ensemble.eval.jepa_harness \
  --llm Qwen/Qwen2.5-0.5B-Instruct \
  --qa research/data/benchmark-qa.jsonl \
  --kb research/data/benchmark-kb.jsonl \
  --limit 50 --n_drafts 8
```

### World-model energy selector

```bash
uv run --package ensemble python -m ensemble.eval.world_harness \
  --llm tiny --toy --limit 20 --n_drafts 8

uv run --package ensemble python -m ensemble.eval.world_harness \
  --llm Qwen/Qwen2.5-0.5B-Instruct \
  --qa research/data/benchmark-qa.jsonl \
  --kb research/data/benchmark-kb.jsonl \
  --limit 50 --n_drafts 8
```

## Layout

```
research/ensemble/
  src/ensemble/
    backends.py       # TinyBackend, HFBackend, TinyLLM, HFLLM
    memory.py         # Embedder, VectorStore, Router
    jepa.py           # JEPA latent predictor
    bridge.py         # LLM hidden -> latent alignment
    world_model.py    # Latent dynamics + rollout
    energy.py         # Energy-based critic
    jepa_ensemble.py  # Ensemble (JEPA track)
    world_ensemble.py # WorldEnsemble
    eval/
      metrics.py
      jepa_harness.py
      world_harness.py
```
