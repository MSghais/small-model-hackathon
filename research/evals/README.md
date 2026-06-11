# SLM Agentic Benchmark Suite

A uv workspace package to evaluate **local HuggingFace models** against agentic benchmarks:

| Benchmark | What it measures | Dataset |
|---|---|---|
| **BFCL v4** | Function-calling accuracy | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` |
| **τ-bench** | Tool-agent-user multi-turn interaction | `ShishirPatil/tau-bench` |
| **GAIA** | General end-to-end agent tasks | `gaia-benchmark/GAIA` |
| **SWE-bench Verified** | Agentic code patching | `princeton-nlp/SWE-bench_Verified` |

## Install

From the repo root:

```bash
uv sync --group evals
```

## Quickstart

```bash
# From repo root (recommended)
uv run --package slm-evals slm-benchmark \
  --model openbmb/MiniCPM5-1B \
  --benchmarks bfcl tau_bench \
  --max-samples 20

# Or as a module
uv run --package slm-evals python -m slm_evals.run_benchmark \
  --model openbmb/MiniCPM5-1B \
  --benchmarks bfcl tau_bench \
  --max-samples 20

# YAML config
uv run --package slm-evals slm-benchmark \
  --config research/evals/configs/experiment_001.yaml
```

## Project structure

```
research/evals/
├── pyproject.toml
├── configs/
│   └── experiment_001.yaml
├── src/slm_evals/
│   ├── run_benchmark.py
│   ├── benchmarks/
│   │   ├── base.py
│   │   ├── bfcl.py
│   │   ├── tau_bench.py
│   │   ├── gaia.py
│   │   └── swe_bench.py
│   └── utils/
│       ├── model_loader.py
│       ├── reporter.py
│       └── config_loader.py
└── results/              # created at runtime (relative to cwd)
```

## CLI reference

```
--model          Path to local HF model dir (or Hub ID)
--benchmarks     Space-separated: bfcl tau_bench gaia swe_bench all
--config         YAML config file (overrides CLI flags)
--max-samples    Cap samples per benchmark
--output-dir     Results directory (default: ./results)
--experiment-name  Tag for this run
--device         auto | cpu | cuda | cuda:0
--dtype          float32 | float16 | bfloat16 | int8 | int4
--max-new-tokens Max tokens per generation (default: 512)
--temperature    Sampling temp (default: 0.0 = greedy)
```

## Adding a custom benchmark

1. Create `src/slm_evals/benchmarks/my_bench.py` and subclass `BaseBenchmark`.
2. Register it in `src/slm_evals/run_benchmark.py` → `BENCHMARK_REGISTRY`.
3. Run: `uv run --package slm-evals slm-benchmark --model ./my-model --benchmarks my_bench`

## Output formats

Results are written under `<output-dir>/<experiment_name>/`:

- `results.json` — full structured dump
- `results.csv` — one row per benchmark
- `report.md` — human-readable summary

## Notes

**τ-bench user simulator**: Default is a lightweight rule-based simulator. Set `use_llm_user: true` in config for the GPT-4o user agent (API cost).

**SWE-bench full eval**: Set `full_eval: true` to run the official Docker harness (`pip install swebench docker`).

**GAIA tools**: Offline by default (`tool_mode: describe`). Wire real tools in `gaia.py` for live eval.
