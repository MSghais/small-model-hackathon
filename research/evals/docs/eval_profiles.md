# Eval profiles guide

Match what your model is supposed to improve to the right benchmark profile, then run it with one command.

Catalog file: [`configs/eval_profiles.yaml`](../configs/eval_profiles.yaml)

## Quick commands

```bash
# See all lm-eval profiles (reasoning, code, smoke, …)
uv run --package slm-evals slm-lm-eval --list-profiles

# Include agentic suites (slm-benchmark) and external notes
uv run --package slm-evals slm-lm-eval --list-profiles-all

# List lm-eval task names available in the harness
uv run --package slm-evals slm-lm-eval --list-tasks

# Agentic benchmark keys (BFCL, τ-bench, GAIA, SWE)
uv run --package slm-evals slm-benchmark --list-benchmarks

# Run a profile by name
uv run --package slm-evals slm-lm-eval \
  --profile reasoning \
  --preset minicpm5-1b \
  --experiment-name minicpm5-1b__reasoning-baseline
```

Install lm-eval extras first: `uv sync --group lm-eval`

---

## Three eval systems in this repo

| System | CLI | What it measures |
| ------ | --- | ---------------- |
| Academic (lm-eval harness) | `slm-lm-eval` | ARC, GSM8K, HumanEval, IFEval, … |
| Agentic | `slm-benchmark` | Function calling, multi-turn tools, GAIA, SWE |
| Ensemble-specific | `jepa_harness`, `world_harness` | JEPA draft selection, world-model energy ranking |

Use **one profile per claim**. Do not compare training loss to lm-eval accuracy.

---

## Match your claim → profile

| If you claim… | Profile / suite | Tool | Tasks or benchmarks |
| ------------- | ----------------- | ---- | ------------------- |
| Quick sanity check | `smoke` | `slm-lm-eval` | `arc_easy`, `hellaswag` (limit 25) |
| Better reasoning | `reasoning` | `slm-lm-eval` | `gsm8k`, `arc_easy`, `arc_challenge`, `hellaswag` |
| Better language understanding | `understanding` | `slm-lm-eval` | `boolq`, `piqa`, `copa`, `rte` |
| Better code generation | `code` | `slm-lm-eval` | `humaneval`, `mbpp` |
| Better instruction following | `instructions` | `slm-lm-eval` | `ifeval` |
| Better French / translation | `french` | `slm-lm-eval` | `french_bench_xnli`, `belebele_fra_Latn`, `wmt14-en-fr`, … |
| Better multilingual understanding | `multilingual` | `slm-lm-eval` | `xnli`, `xcopa`, `xwinograd` |
| General ~1B SLM baseline | `general_slm` | `slm-lm-eval` | 6-task mix (full splits) |
| Baseline vs finetune study | `compare_study` | `slm-lm-eval` | Same 6 tasks, limit 100 |
| Tool use / function calling | `agentic_tool_use` | `slm-benchmark` | `bfcl`, `tau_bench` |
| End-to-end assistant tasks | `agentic_gaia` | `slm-benchmark` | `gaia` |
| Real-world code repair | `agentic_code` | `slm-benchmark` | `swe_bench` |
| JEPA / selector quality | `jepa_selector` | `jepa_harness` | Domain QA + draft ablations |
| World model / planning | `world_model` | `world_harness` | Energy-ranked drafts on QA |
| Better embeddings | `embeddings_mteb` | external (MTEB) | Not in this repo |
| Chat quality (judge-based) | `chat_judge` | external | MT-Bench, AlpacaEval |

---

## Profile YAML files

| Profile key | Config file |
| ----------- | ----------- |
| `smoke` | `lm_eval_smoke.yaml` |
| `reasoning` | `lm_eval_reasoning.yaml` |
| `understanding` | `lm_eval_understanding.yaml` |
| `code` | `lm_eval_code.yaml` |
| `instructions` | `lm_eval_instructions.yaml` |
| `french` | `lm_eval_french.yaml` |
| `multilingual` | `lm_eval_multilingual.yaml` |
| `general_slm` | `lm_eval_minicpm5.yaml` |
| `compare_study` | `lm_eval_compare_study.yaml` |

Equivalent to `--profile reasoning`:

```bash
uv run --package slm-evals slm-lm-eval \
  --config research/evals/configs/lm_eval_reasoning.yaml \
  --preset minicpm5-1b
```

---

## Baseline vs candidate workflow

Use the **same profile** for both runs; only change preset and experiment name:

```bash
PROFILE=reasoning
BASE=minicpm5-1b__reasoning-baseline
CAND=minicpm5-1b-lora__reasoning

uv run --package slm-evals slm-lm-eval \
  --profile "$PROFILE" --preset minicpm5-1b --experiment-name "$BASE"

uv run --package slm-evals slm-lm-eval \
  --profile "$PROFILE" --preset minicpm5-1b-lesson-lora \
  --experiment-name "$CAND" \
  --compare-to "results/lm_eval/${BASE}/results.json"
```

Or after finetune:

```bash
uv run python research/finetune.py --preset minicpm5-1b --mode lora \
  --lm-eval-after \
  --lm-eval-config research/evals/configs/lm_eval_reasoning.yaml \
  --lm-eval-baseline minicpm5-1b
```

---

## Results layout

**slm-lm-eval** → `results/lm_eval/<experiment-name>/`

| File | Contents |
| ---- | -------- |
| `results.json` | Full lm-eval payload + `run_meta` |
| `summary.md` | Task → metric table |
| `run_meta.json` | Profile tasks, preset, seed |
| `comparison.md` | Delta vs baseline (with `--compare-to`) |

**slm-benchmark** → `results/<experiment-name>/` (`results.json`, `results.csv`, `report.md`)

---

## Custom tasks

Override tasks on any profile:

```bash
uv run --package slm-evals slm-lm-eval \
  --profile smoke \
  --tasks gsm8k arc_easy \
  --preset minicpm5-1b
```

Browse all harness tasks: `slm-lm-eval --list-tasks-all`

See also: [USAGE.md](../USAGE.md), [benchmarks.md](benchmarks.md)
