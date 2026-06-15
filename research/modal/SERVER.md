# GPU worker runbook (`server_app.py`)

Long-lived Modal GPU for iterative finetune / eval loops. Intended for **humans** and **AI coding agents** running many experiments from the same warm container.

**Full docs:** [README.md](README.md) · **Code:** [`server_app.py`](server_app.py) · **Jobs:** [`experiments.yaml`](experiments.yaml)

---

## Prerequisites

Run from **repo root**.

```bash
pip install modal
modal setup
modal secret create huggingface HF_TOKEN=<your-hf-token>   # once
modal deploy research/modal/server_app.py                 # once per image change
```

| Name | Value |
| ---- | ----- |
| App | `slm-gpu-worker` |
| Class | `GpuWorker` |
| GPU | `A10G` |
| Volumes | `hf-cache` → `/root/.cache/huggingface`, `slm-finetune` → `/vol/finetuned` |

---

## Start session (human or agent)

```bash
# Option A: block terminal (default 4h keep-alive)
modal run research/modal/server_app.py

# Option B: detached — preferred for agent loops
modal run -d research/modal/server_app.py --hours 6

# Verify worker
modal run research/modal/server_app.py --ping
# → {"status": "ok", "app": "slm-gpu-worker"}
```

---

## Experiment commands (repeat freely)

All commands use the deployed warm worker when `modal deploy` has been run.

```bash
# --- Train ---
modal run research/modal/server_app.py --job lesson-lora --max-steps 20
modal run research/modal/server_app.py --job alpaca-lora --max-steps 50
modal run research/modal/server_app.py --job smoltalk-lora --max-steps 50

# --- Eval only (adapter must exist on Volume) ---
modal run research/modal/server_app.py --eval-only --job lesson-lora
modal run research/modal/server_app.py --eval-only   # all jobs in experiments.yaml

# --- Full pipeline (same container: baseline → train → eval) ---
modal run research/modal/server_app.py --pipeline --job lesson-lora --max-steps 20
modal run research/modal/server_app.py --pipeline --job lesson-lora --max-steps 20 --skip-baseline

# --- Custom finetune.py flags ---
modal run research/modal/server_app.py --cmd \
  "uv run python research/finetune.py --preset minicpm5-1b --mode lora \
   --dataset research/data/education-lesson-chat.jsonl --format chat \
   --out /vol/finetuned/lesson-lora --max_steps 10"

# --- Custom lm-eval ---
modal run research/modal/server_app.py --cmd \
  "uv run --package slm-evals slm-lm-eval \
   --config research/evals/configs/lm_eval_smoke.yaml \
   --experiment-name lesson-lora__manual \
   --output-dir /vol/finetuned/results/lm_eval \
   --model openbmb/MiniCPM5-1B \
   --adapter /vol/finetuned/lesson-lora"
```

Job names and datasets: [`experiments.yaml`](experiments.yaml).

---

## Inspect results (human or agent)

```bash
# List Volume
modal volume ls slm-finetune
modal volume ls slm-finetune lesson-lora
modal volume ls slm-finetune results/lm_eval

# Download to laptop
modal volume get slm-finetune lesson-lora ./models/finetuned/minicpm5-1b-lora
modal volume get slm-finetune results/lm_eval ./results/lm_eval

# Stream worker logs
modal app logs slm-gpu-worker -f
```

Key artifacts on Volume:

| Path | Content |
| ---- | ------- |
| `/vol/finetuned/<job>/` | LoRA adapter + `training_results.json` |
| `/vol/finetuned/results/lm_eval/<exp>/` | `results.json`, `summary.md`, `comparison.md` |

---

## End session

```bash
modal app stop slm-gpu-worker -y
```

Stops the deployed app and warm GPU pool. Volume data is retained.

---

## AI agent loop (structured)

Use this sequence when an agent is iterating on training or eval without local CUDA.

```
1. CHECK   modal run research/modal/server_app.py --ping
2. BOOT    if ping fails → modal deploy ... then modal run -d ... --hours 6
3. SMOKE   modal run ... --job lesson-lora --max-steps 5
4. EVAL    modal run ... --eval-only --job lesson-lora
5. READ    modal volume ls slm-finetune results/lm_eval
           modal volume get ... (or read comparison.md locally after get)
6. ADJUST  edit experiments.yaml OR pass --max-steps / --lm-eval-config
7. GOTO 3  until metrics acceptable
8. PULL    modal volume get slm-finetune lesson-lora ./models/finetuned/minicpm5-1b-lora
9. STOP    modal app stop slm-gpu-worker -y   (optional, saves GPU cost)
```

### Agent decision rules

| Situation | Action |
| --------- | ------ |
| First time in repo | `modal deploy research/modal/server_app.py` |
| `ping` returns ok | Skip boot; run task commands |
| `ping` fails / timeout | `modal run -d research/modal/server_app.py --hours 6`, retry ping |
| Train OOM | `--cmd` with `--mode qlora` or lower `--max-steps` |
| Eval missing adapter | Train first, or `modal volume ls slm-finetune <job>` |
| Need batch parallel GPUs | Use `finetune_app.py --parallel` instead |
| Need one-shot CI sweep | Use `finetune_app.py` (not server) |
| Image / code changed | Re-run `modal deploy research/modal/server_app.py` |

### Python API (agents in Modal notebook or scripts)

```python
import modal

Worker = modal.Cls.from_name("slm-gpu-worker", "GpuWorker")
w = Worker()

assert w.ping.remote()["status"] == "ok"

w.finetune.remote({
    "name": "lesson-lora",
    "preset": "minicpm5-1b",
    "mode": "lora",
    "dataset": "research/data/education-lesson-chat.jsonl",
    "format": "chat",
    "max_steps": 20,
})

w.run_pipeline.remote(job_names=["lesson-lora"], max_steps=20)
```

---

## `finetune_app.py` vs `server_app.py`

| | `finetune_app.py` | `server_app.py` |
| --- | --- | --- |
| App name | `slm-finetune-benchmark` | `slm-gpu-worker` |
| Container | New per function call | Warm pool, reused |
| Deploy | Optional | **Required** for cross-terminal reuse |
| Parallel jobs | `--parallel` (3 GPUs) | Sequential on one GPU |
| Best for | Full sweep, reproducible batch | Interactive / agent iteration |
| Entry | `modal run research/modal/finetune_app.py` | `modal deploy` + `modal run research/modal/server_app.py` |

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `scaledown_window must be between 2 and 3600` | Already fixed in `_common.py` (3600 max) |
| Deploy succeeds but ping fails | Wait ~30s for warm pool; check `modal app list` |
| Command uses cold container | Run `modal deploy` first; confirm app name `slm-gpu-worker` |
| HF download every run | `hf-cache` volume should mount; first run populates cache |
| Writes not visible | Paths must be under `/vol/finetuned/`, not `/repo/models/` |
| GPU still billing overnight | `modal app stop slm-gpu-worker` |

---

## References

- [Modal Volumes](https://modal.com/docs/guide/volumes)
- [Modal Images](https://modal.com/docs/guide/images)
- [modal run](https://modal.com/docs/reference/cli/run)
- [modal app stop](https://modal.com/docs/reference/cli/app#modal-app-stop)
- [modal shell](https://modal.com/docs/reference/cli/shell) — debug: `modal shell research/modal/server_app.py::GpuWorker.finetune`
