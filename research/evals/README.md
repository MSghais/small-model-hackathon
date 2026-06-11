# SLM Agentic Benchmark Suite

A self-contained Python toolkit to evaluate **local HuggingFace models**
against the four canonical small-model agentic benchmarks:

| Benchmark | What it measures | Dataset |
|---|---|---|
| **BFCL v4** | Function-calling accuracy (single & multi-turn) | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` |
| **τ-bench** | Tool-agent-user multi-turn interaction | `ShishirPatil/tau-bench` |
| **GAIA** | General end-to-end agent tasks | `gaia-benchmark/GAIA` |
| **SWE-bench Verified** | Agentic code patching | `princeton-nlp/SWE-bench_Verified` |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all benchmarks against a local model (smoke test: 20 samples each)
python run_benchmark.py \
  --model ./path/to/my-model \
  --benchmarks all \
  --max-samples 20

# 3. Run specific benchmarks
python run_benchmark.py \
  --model ./path/to/my-model \
  --benchmarks bfcl tau_bench

# 4. Run from a YAML config (recommended for experiments)
python run_benchmark.py --config configs/experiment_001.yaml
```

---

## Project Structure

```
slm_bench/
├── run_benchmark.py          # Main entrypoint
├── requirements.txt
├── configs/
│   └── experiment_001.yaml   # Template experiment config
├── benchmarks/
│   ├── base.py               # Abstract base class
│   ├── bfcl.py               # BFCL function-calling
│   ├── tau_bench.py          # τ-bench tool-agent-user
│   ├── gaia.py               # GAIA general agent
│   └── swe_bench.py          # SWE-bench coding
├── utils/
│   ├── model_loader.py       # HF Transformers loader
│   ├── reporter.py           # JSON / CSV / Markdown output
│   └── config_loader.py      # YAML + CLI arg handling
└── results/                  # Auto-created output directory
    └── <experiment_name>/
        ├── results.json
        ├── results.csv
        └── report.md
```

---

## CLI Reference

```
--model          Path to local HF model dir (or Hub ID)
--benchmarks     Space-separated list: bfcl tau_bench gaia swe_bench all
--config         YAML config file (overrides all CLI flags)
--max-samples    Cap samples per benchmark (default: all)
--output-dir     Where to write results (default: ./results)
--experiment-name  Tag for this run (auto-generated if omitted)
--device         HF device_map: auto | cpu | cuda | cuda:0
--dtype          float32 | float16 | bfloat16 | int8 | int4
--max-new-tokens Max tokens per generation (default: 512)
--temperature    Sampling temp (default: 0.0 = greedy)
```

---

## Adding a Custom Benchmark

1. Create `benchmarks/my_bench.py` and subclass `BaseBenchmark`:

```python
from benchmarks.base import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    name = "my_bench"

    def load_dataset(self):
        # Return list of sample dicts
        return [{"id": "001", "input": "...", "answer": "..."}]

    def build_prompt(self, sample):
        return f"Answer this: {sample['input']}\nAnswer:"

    def evaluate_sample(self, sample, prediction):
        passed = prediction.strip() == sample["answer"]
        return {"passed": passed, "score": float(passed), "note": ""}
```

2. Register it in `run_benchmark.py`:

```python
from benchmarks.my_bench import MyBenchmark
BENCHMARK_REGISTRY["my_bench"] = MyBenchmark
```

3. Run it:

```bash
python run_benchmark.py --model ./my-model --benchmarks my_bench
```

---

## Output Formats

**JSON** (`results.json`) — full structured dump including per-sample predictions.

**CSV** (`results.csv`) — one row per benchmark; easy to paste into spreadsheets
or compare runs with pandas:
```python
import pandas as pd
df = pd.read_csv("results/my_run/results.csv")
print(df[["benchmark", "score", "avg_latency_s"]])
```

**Markdown** (`report.md`) — human-readable report with score tables
and per-sample pass/fail details.

---

## Notes on Full Evaluation Modes

**τ-bench user simulator**: By default uses a lightweight rule-based simulator.
Set `use_llm_user: true` in config to enable the GPT-4o user agent (matches
the official benchmark protocol but incurs API cost).

**SWE-bench full eval**: Set `full_eval: true` to run the official Docker harness
and execute actual test suites. Requires:
```bash
pip install swebench docker
# and Docker running locally
```

**GAIA tools**: By default, tools are *described* in the prompt but not executed
(offline eval). Wire in real `web_search` / `code_exec` callables in
`benchmarks/gaia.py → _run_episode()` for live agentic evaluation.
