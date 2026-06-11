# Ensemble research package

JEPA and world-model ensemble experiments. Stays under `research/` — not wired into the Gradio agent.

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
