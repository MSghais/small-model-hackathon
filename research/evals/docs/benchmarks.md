# Benchmark reference

What each benchmark in `slm-evals` measures, where data comes from, and how to configure overrides.

All benchmarks extend `BaseBenchmark` (`src/slm_evals/benchmarks/base.py`):

1. `load_dataset()` — fetch samples (Hub or local JSONL)
2. `build_prompt(sample)` — format the model input
3. `evaluate_sample(sample, prediction)` — return `{passed, score, note}`
4. `run()` — iterate, call `generate_fn`, aggregate scores (inherited)

---

## Summary table

| Key | Benchmark | Measures | Default dataset |
| --- | --------- | -------- | --------------- |
| `bfcl` | Berkeley Function-Calling Leaderboard v4 | Single-turn function call accuracy | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` |
| `tau_bench` | τ-bench | Multi-turn tool + user simulation | `ShishirPatil/tau-bench` |
| `gaia` | GAIA | End-to-end agent tasks (reasoning + tools) | `gaia-benchmark/GAIA` |
| `swe_bench` | SWE-bench Verified | Code patch generation for real issues | `princeton-nlp/SWE-bench_Verified` |

---

## BFCL (`bfcl`)

**Goal:** Given a user request and a function schema, does the model emit a valid JSON tool call with the correct name and arguments?

**Prompt style:** System message lists available functions; model must reply with only:

```json
{"name": "<function_name>", "arguments": {<key>: <value>}}
```

**Scoring:**

- Function name must match exactly
- Arguments: exact match if `strict: true`, fuzzy match if `strict: false` (recommended for SLMs)

**Config overrides** (`benchmark_overrides.bfcl`):

| Key | Default | Description |
| --- | ------- | ----------- |
| `data_path` | Hub | Local JSONL instead of Hub download |
| `categories` | `[]` (all) | Filter BFCL categories |
| `strict` | `false` | Require perfect argument match |

**Implementation:** `src/slm_evals/benchmarks/bfcl.py`

---

## τ-bench (`tau_bench`)

**Goal:** Multi-turn dialogue where the model acts as a tool-using agent while a simulated user drives the conversation toward a goal (e.g. retail order change).

**Scoring:** Task success after up to `max_turns` exchanges — did the agent satisfy the user's underlying intent using the right tools?

**Config overrides** (`benchmark_overrides.tau_bench`):

| Key | Default | Description |
| --- | ------- | ----------- |
| `data_path` | Hub | Local JSONL |
| `domain` | `retail` | `retail`, `airline`, or `both` |
| `max_turns` | `15` | Dialogue cap |
| `use_llm_user` | `false` | `true` → GPT-4o user simulator (paid API) |

**Notes:**

- Default user simulator is rule-based — no API key required
- Small models often struggle on long horizons; start with `--max-samples 10`

**Implementation:** `src/slm_evals/benchmarks/tau_bench.py`

---

## GAIA (`gaia`)

**Goal:** Real-world assistant tasks requiring reasoning, optional tool use, and concise final answers (web search, files, calculation, etc.).

**Prompt style:** Question + level metadata; tool availability depends on `tool_mode`.

**Scoring:** Normalized answer match against GAIA reference (with level breakdown in aggregates).

**Config overrides** (`benchmark_overrides.gaia`):

| Key | Default | Description |
| --- | ------- | ----------- |
| `data_path` | Hub | Local JSONL |
| `split` | `validation` | Public `validation`; `test` may need HF auth |
| `levels` | `[1, 2]` | Difficulty levels 1–3 |
| `tool_mode` | `describe` | `describe` = offline tool docs; `none` = no tools |

**Notes:**

- `tool_mode: describe` does not execute live tools — suitable for offline SLM scoring
- For live tool eval, extend `gaia.py` with real tool backends

**Implementation:** `src/slm_evals/benchmarks/gaia.py`

---

## SWE-bench Verified (`swe_bench`)

**Goal:** Given a GitHub issue and codebase context, produce a unified diff that fixes the bug.

**Modes:**

| `full_eval` | Behavior |
| ----------- | -------- |
| `false` (default) | Generate patch text; score with lightweight heuristics / match checks — no Docker |
| `true` | Official SWE-bench harness — runs tests in containers (`swebench` + Docker) |

**Config overrides** (`benchmark_overrides.swe_bench`):

| Key | Default | Description |
| --- | ------- | ----------- |
| `data_path` | Hub | Local JSONL |
| `full_eval` | `false` | Enable Docker harness |
| `context_lines` | `80` | Surrounding code context in prompt |

**Notes:**

- Full eval is slow and resource-heavy — use for final validation only
- SLMs typically score low; use `--max-samples` for iterative prompt tuning

**Implementation:** `src/slm_evals/benchmarks/swe_bench.py`

---

## Model loading

Shared loader: `src/slm_evals/utils/model_loader.py`

Returns a `model_bundle` dict passed to each benchmark:

- `generate_fn(prompt, max_new_tokens, temperature)` — unified generation interface
- `param_count` — billions of parameters (for reporting)
- Underlying `model` / `tokenizer` handles

Quantization (`int8`, `int4`) uses `bitsandbytes` when available.

---

## Reporter output schema

`Reporter.save()` (`src/slm_evals/utils/reporter.py`) writes:

**Per benchmark in JSON:**

```json
{
  "name": "bfcl",
  "total": 100,
  "passed": 42,
  "score": 0.42,
  "samples": [...]
}
```

**Aggregate fields:**

- `experiment_name`, `model_path`, `timestamp`
- `aggregate_score` — mean of benchmark scores

CSV columns: `benchmark`, `total`, `passed`, `score`.
