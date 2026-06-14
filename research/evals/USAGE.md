# Evals usage

Run the **SLM Agentic Benchmark Suite** (`slm-evals`) against a local HuggingFace model directory or Hub id.

Benchmark details: [docs/benchmarks.md](docs/benchmarks.md). Package overview: [README.md](README.md).

## Install

From the repo root:

```bash
uv sync --group evals
```

For academic benchmarks (lm-evaluation-harness):

```bash
uv sync --group lm-eval
```

This installs the `slm-evals` workspace package and registers the `slm-benchmark` and `slm-lm-eval` console scripts.

## Quick start

```bash
# Two benchmarks, capped samples (good first run)
uv run --package slm-evals slm-benchmark \
  --model openbmb/MiniCPM5-1B \
  --benchmarks bfcl tau_bench \
  --max-samples 20

# All four benchmarks
uv run --package slm-evals slm-benchmark \
  --model ./models/finetuned/minicpm5-1b-lora \
  --benchmarks all \
  --max-samples 50

# Equivalent module invocation
uv run --package slm-evals python -m slm_evals.run_benchmark \
  --model openbmb/MiniCPM5-1B \
  --benchmarks bfcl \
  --max-samples 10
```

## Config-driven runs

Copy and edit the template, then pass `--config`:

```bash
cp research/evals/configs/experiment_001.yaml research/evals/configs/my_run.yaml
# edit model_path, benchmarks, max_samples, overrides

uv run --package slm-evals slm-benchmark \
  --config research/evals/configs/my_run.yaml
```

When `--config` is set, **YAML values override CLI flags**. Use configs for reproducible experiment names and per-benchmark settings.

### Template fields

| Key | Description |
| --- | ----------- |
| `model_path` | Local directory or HF Hub id |
| `device` | `auto`, `cpu`, `cuda`, `cuda:0`, … |
| `dtype` | `float32`, `float16`, `bfloat16`, `int8`, `int4` |
| `max_new_tokens` | Cap per generation (default 512) |
| `temperature` | `0.0` = greedy (recommended for evals) |
| `experiment_name` | Folder name under `output_dir` |
| `output_dir` | Root for results (default `results`) |
| `benchmarks` | List: `bfcl`, `tau_bench`, `gaia`, `swe_bench` |
| `max_samples` | Cap per benchmark; omit or `null` for full split |
| `benchmark_overrides` | Per-benchmark dict (see [docs/benchmarks.md](docs/benchmarks.md)) |

---

## CLI reference

```
slm-benchmark [OPTIONS]

--list-benchmarks       Show agentic benchmark keys and preset suites
--model PATH            Local HF dir or Hub id (required unless --config)
--benchmarks NAMES      bfcl tau_bench gaia swe_bench all  (default: all)
--config PATH           YAML config (overrides other flags)
--max-samples N         Cap samples per benchmark
--output-dir DIR        Results root (default: ./results)
--experiment-name TAG   Run folder name (auto timestamp if omitted)
--device MAP            auto | cpu | cuda | cuda:0
--dtype TYPE            float32 | float16 | bfloat16 | int8 | int4
--max-new-tokens N      Default 512
--temperature T         Default 0.0
```

---

## Results

Each run writes to `<output_dir>/<experiment_name>/`:

| File | Contents |
| ---- | -------- |
| `results.json` | Full structured payload (per-sample + aggregates) |
| `results.csv` | One row per benchmark |
| `report.md` | Human-readable summary |

Example layout:

```text
results/
└── minicpm5-1b__bfcl-tau__v1/
    ├── results.json
    ├── results.csv
    └── report.md
```

`output_dir` is relative to **current working directory**. Run from repo root so paths stay predictable, or set an absolute `output_dir` in YAML.

---

## Per-benchmark tips

### BFCL (function calling)

- Default: downloads from `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- `strict: false` in YAML — fuzzy argument matching (better for small models)
- Local JSONL: set `benchmark_overrides.bfcl.data_path`

### τ-bench (multi-turn tools)

- Domains: `retail`, `airline`, or `both`
- `use_llm_user: false` — free rule-based user simulator (default)
- `use_llm_user: true` — GPT-4o user agent (**API cost**)

### GAIA

- Default split: `validation` (public)
- `tool_mode: describe` — offline tool descriptions (no live web)
- Level filter: `levels: [1, 2]` or `[1, 2, 3]`

### SWE-bench Verified

- Default: lightweight patch-generation scoring (no Docker)
- `full_eval: true` — official harness (`pip install swebench docker`)

See [docs/benchmarks.md](docs/benchmarks.md) for scoring semantics.

---

## lm-evaluation-harness (`slm-lm-eval`)

Run standard academic benchmarks (ARC, HellaSwag, PIQA, BoolQ, GSM8K) via [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Install: `uv sync --group lm-eval`

Full profile guide: [docs/eval_profiles.md](docs/eval_profiles.md)

### Discover profiles and tasks

```bash
# Claim-matched lm-eval profiles (reasoning, code, smoke, …)
uv run --package slm-evals slm-lm-eval --list-profiles

# Also show agentic suites + external benchmark notes
uv run --package slm-evals slm-lm-eval --list-profiles-all

# lm-eval task names
uv run --package slm-evals slm-lm-eval --list-tasks

# Agentic benchmarks (BFCL, τ-bench, GAIA, SWE)
uv run --package slm-evals slm-benchmark --list-benchmarks
```

### Quick start

```bash
# By profile name (recommended)
uv run --package slm-evals slm-lm-eval \
  --profile reasoning \
  --preset minicpm5-1b \
  --experiment-name minicpm5-1b__reasoning-baseline

# Smoke profile (25 samples)
uv run --package slm-evals slm-lm-eval \
  --profile smoke \
  --preset minicpm5-1b \
  --experiment-name minicpm5-1b__smoke

# LoRA adapter via preset (base + peft resolved automatically)
uv run --package slm-evals slm-lm-eval \
  --config research/evals/configs/lm_eval_minicpm5.yaml \
  --preset minicpm5-1b-lesson-lora \
  --experiment-name minicpm5-1b-lora__v1

# Explicit base + adapter
uv run --package slm-evals slm-lm-eval \
  --config research/evals/configs/lm_eval_smoke.yaml \
  --model openbmb/MiniCPM5-1B \
  --adapter ./models/finetuned/minicpm5-1b-lora \
  --experiment-name minicpm5-1b-lora__manual
```

### Compare baseline vs candidate

Use the **same config** for both runs; only change `--preset` / `--experiment-name`:

```bash
uv run --package slm-evals slm-lm-eval \
  --config research/evals/configs/lm_eval_compare_study.yaml \
  --preset minicpm5-1b \
  --experiment-name minicpm5-1b__baseline

uv run --package slm-evals slm-lm-eval \
  --config research/evals/configs/lm_eval_compare_study.yaml \
  --preset minicpm5-1b-lesson-lora \
  --experiment-name minicpm5-1b-lora__v1 \
  --compare-to results/lm_eval/minicpm5-1b__baseline/results.json
```

### Config templates

Catalog: `configs/eval_profiles.yaml` — maps **claim → profile → tasks**.

| Profile (`--profile`) | Config file | Purpose |
| --------------------- | ----------- | ------- |
| `smoke` | `lm_eval_smoke.yaml` | Fast validation (`limit: 25`, 2 tasks) |
| `reasoning` | `lm_eval_reasoning.yaml` | Math + commonsense (GSM8K, ARC, HellaSwag) |
| `understanding` | `lm_eval_understanding.yaml` | NLU (BoolQ, PIQA, COPA, RTE) |
| `code` | `lm_eval_code.yaml` | HumanEval + MBPP |
| `instructions` | `lm_eval_instructions.yaml` | IFEval instruction following |
| `general_slm` | `lm_eval_minicpm5.yaml` | Full ~1B SLM profile (6 tasks) |
| `compare_study` | `lm_eval_compare_study.yaml` | Baseline vs finetune comparison defaults |

| Key | Description |
| --- | ----------- |
| `tasks` | lm-eval task names (e.g. `arc_easy`, `gsm8k`) |
| `num_fewshot` | Few-shot count (gsm8k may use task default 8) |
| `limit` | Max samples per task; `null` = full split |
| `seed` | Random seed (applied to all lm-eval RNGs) |
| `batch_size` | `auto` or integer |
| `device` | `auto`, `cpu`, `cuda`, … |
| `dtype` | `bfloat16`, `float16`, `int4`, … |
| `trust_remote_code` | Required for MiniCPM / Gemma presets |
| `output_dir` | Root for runs (default `results/lm_eval`) |

### CLI reference

```
slm-lm-eval [OPTIONS]

--list-profiles         Show claim-matched profiles and example commands
--list-profiles-all     Include agentic suites and external benchmark notes
--list-tasks            List lm-eval task names (catalog fallback if not installed)
--list-tasks-all        Full lm-eval task list
--profile NAME          Shorthand for --config (reasoning, code, smoke, …)
--config PATH           YAML config (tasks, seed, limit, …)
--preset KEY            models.yaml preset (base, LoRA, merged)
--model PATH            HF Hub id or merged checkpoint dir
--adapter PATH          LoRA adapter (alternative to preset adapter_path)
--tasks NAMES           Override task list
--num-fewshot N
--limit N               Cap samples per task
--seed N
--batch-size VALUE
--device MAP
--dtype TYPE
--output-dir DIR        Default: results/lm_eval
--experiment-name TAG   Run folder name
--compare-to PATH       Baseline results.json for delta table
```

### Results

Each run writes to `<output_dir>/<experiment_name>/`:

| File | Contents |
| ---- | -------- |
| `results.json` | lm-eval native payload + `run_meta` |
| `summary.md` | Task → metric table |
| `run_meta.json` | Preset, base model, adapter, tasks, seed |
| `comparison.md` | Delta table (when `--compare-to` set) |

### PEFT / LoRA

lm-eval expects `pretrained=<base>,peft=<adapter>`. The preset resolver handles this for keys like `minicpm5-1b-lesson-lora`. Merged checkpoints use `--preset minicpm5-1b-lesson-merged` or `--model ./models/finetuned/...-merged`.

---

## Adding a custom benchmark

1. Create `src/slm_evals/benchmarks/my_bench.py` subclassing `BaseBenchmark`:
   - `load_dataset()` → list of sample dicts
   - `build_prompt(sample)` → prompt string
   - `evaluate_sample(sample, prediction)` → `{passed, score, note}`

2. Register in `src/slm_evals/run_benchmark.py` → `BENCHMARK_REGISTRY`.

3. Run:
   ```bash
   uv run --package slm-evals slm-benchmark \
     --model ./my-model --benchmarks my_bench --max-samples 10
   ```

---

## Suggested workflows

### Smoke (CPU/GPU, ~5 min)

```bash
uv run --package slm-evals slm-benchmark \
  --model openbmb/MiniCPM5-1B \
  --benchmarks bfcl \
  --max-samples 5 \
  --device cpu
```

### Before / after fine-tune

```bash
BASE=openbmb/MiniCPM5-1B
ADAPTER=./models/finetuned/minicpm5-1b-lora

for M in "$BASE" "$ADAPTER"; do
  uv run --package slm-evals slm-benchmark \
    --model "$M" \
    --benchmarks bfcl tau_bench \
    --max-samples 100 \
    --experiment-name "$(basename "$M")__bfcl-tau"
done
```

### Full experiment (YAML)

Edit `configs/experiment_001.yaml` with your `model_path` and `experiment_name`, then:

```bash
uv run --package slm-evals slm-benchmark \
  --config research/evals/configs/experiment_001.yaml
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `error: --model is required` | No `--config` and no `--model` | Pass one of them |
| CUDA OOM | Model too large for VRAM | `--dtype int4` or `--device cpu` |
| HF dataset 401 on GAIA test | Gated split | Use `split: validation` |
| τ-bench hangs / costs | LLM user enabled | Set `use_llm_user: false` |
| Empty `results/` | Wrong cwd | Run from repo root or use absolute `output_dir` |
| Import errors | Evals group not synced | `uv sync --group evals` |

---

## Entry points

| Path | Role |
| ---- | ---- |
| `slm-benchmark` | Agentic benchmarks (BFCL, τ-bench, GAIA, SWE) |
| `slm-lm-eval` | Academic benchmarks via lm-evaluation-harness |
| `python -m slm_evals.run_benchmark` | Same as `slm-benchmark` |
| `python -m slm_evals.run_lm_eval` | Same as `slm-lm-eval` |
| `research/evals/run_benchmark.py` | Thin shim for backward compatibility |
