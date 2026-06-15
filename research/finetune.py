"""
finetune.py — Fine-tune a small LLM: FULL, LoRA, or QLoRA, one script.
======================================================================

Install:
    pip install torch transformers datasets peft accelerate
    pip install bitsandbytes        # only needed for --mode qlora

Model resolution (first match wins)
------------------------------------
1. --model <hf-id-or-path>
2. --preset <key> from models.yaml (or FINETUNE_PRESET env)
3. MODEL_ID / BASE env (raw Hugging Face id or local path)
4. ACTIVE_MODEL preset from models.yaml (text transformers presets only)

Outputs are saved under ./models/finetuned/<preset>-<mode>/ by default.

Examples
--------
# LoRA on the lesson-agent chat dataset using models.yaml preset
python research/finetune.py --preset minicpm5-1b --mode lora --epochs 1

# Same, but read ACTIVE_MODEL / BASE from .env (auto-loaded from repo root)
python research/finetune.py --mode lora --max_steps 50

# LoRA on an instruction dataset from the Hub
python research/finetune.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset tatsu-lab/alpaca --format alpaca \
    --mode lora --epochs 1

# QLoRA (4-bit) on a local JSONL chat file: {"messages": [{"role":..,"content":..}, ...]}
python research/finetune.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset ./data/chats.jsonl --format chat \
    --mode qlora

# Hugging Face Hub datasets (--dataset is the repo id; optional --dataset-config / --split)
python research/finetune.py \
    --preset minicpm5-1b --mode qlora \
    --dataset tatsu-lab/alpaca --format alpaca --dataset-split train

python research/finetune.py \
    --preset minicpm5-1b --mode lora \
    --dataset HuggingFaceTB/smoltalk --format chat \
    --dataset-config all --dataset-split train[:500]

# Env vars also work: FINETUNE_DATASET, FINETUNE_DATASET_CONFIG, FINETUNE_DATASET_SPLIT

# FULL fine-tune on raw text files (continued pretraining style)
python research/finetune.py \
    --model HuggingFaceTB/SmolLM2-360M \
    --dataset ./data/corpus.txt --format text \
    --mode full --lr 2e-5

# After LoRA training, merge adapter into standalone weights:
python research/finetune.py --merge ./models/finetuned/minicpm5-1b-lora \
    --out ./models/finetuned/minicpm5-1b-merged

Dataset formats (--format)
--------------------------
  alpaca : columns instruction / input(optional) / output
  chat   : column  messages = [{"role": "...", "content": "..."}]
  prompt : columns prompt / completion  (or prompt / response)
  text   : column  text  — or a plain .txt file (one doc per line / whole file)

Local files: .json, .jsonl, .csv, .txt. Hub ids: any datasets repo.

Hub datasets useful for the lesson / teacher agent (--format must match columns):
  tatsu-lab/alpaca              alpaca   instruction tuning (general)
  HuggingFaceTB/smoltalk        chat     multi-turn chat (use config: all)
  Open-Orca/OpenOrca            prompt   instruction + response pairs
  databricks/databricks-dolly-15k alpaca   short Q&A, good for small models

After training, metrics are written to <out>/training_results.json
(train/eval loss, perplexity, result_score 0–100).
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _REPO_ROOT / "research/data/education-lesson-chat.jsonl"
_FINETUNE_ROOT = _REPO_ROOT / "models/finetuned"
_FALLBACK_FINETUNE_PRESET = "minicpm5-1b"


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing env vars."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _ensure_repo_on_path() -> None:
    libs = _REPO_ROOT / "libs" / "inference" / "src"
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))


def _is_finetuneable_preset(model) -> bool:
    return model.backend == "transformers" and not model.multimodal and bool(
        model.model_id
    )


def resolve_model_and_preset(
    *,
    model_arg: str | None,
    preset_arg: str | None,
) -> tuple[str, str | None, bool]:
    """Return (model_id_or_path, preset_key, trust_remote_code)."""
    if model_arg:
        trust = os.environ.get("TRUST_REMOTE_CODE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        return model_arg, preset_arg, trust

    for env_name in ("FINETUNE_MODEL", "MODEL_ID", "BASE"):
        raw = os.environ.get(env_name)
        if raw:
            trust = os.environ.get("TRUST_REMOTE_CODE", "").lower() in {
                "1",
                "true",
                "yes",
            }
            return raw, preset_arg, trust

    _ensure_repo_on_path()
    from inference.config import get_app_config, get_model_config

    app_config = get_app_config(reload=True)
    preset_key = (
        preset_arg
        or os.environ.get("FINETUNE_PRESET")
        or os.environ.get("ACTIVE_MODEL")
    )

    if preset_key and preset_key in app_config.models:
        model = get_model_config(preset_key)
        if not _is_finetuneable_preset(model):
            print(
                f"Preset {preset_key!r} is {model.backend}"
                + (" multimodal" if model.multimodal else "")
                + "; falling back to a text transformers preset for fine-tuning."
            )
            preset_key = None

    if preset_key is None:
        for candidate in (_FALLBACK_FINETUNE_PRESET, *app_config.models):
            if candidate not in app_config.models:
                continue
            model = get_model_config(candidate)
            if _is_finetuneable_preset(model):
                preset_key = candidate
                break

    if not preset_key:
        raise SystemExit(
            "No fine-tunable transformers preset found. Pass --model or set BASE/MODEL_ID."
        )

    model = get_model_config(preset_key)
    if not _is_finetuneable_preset(model):
        raise SystemExit(
            f"Preset {preset_key!r} cannot be fine-tuned "
            f"(backend={model.backend}, multimodal={model.multimodal})."
        )
    return model.model_id, preset_key, model.trust_remote_code


def default_output_dir(preset_key: str | None, mode: str) -> str:
    label = preset_key or "custom"
    return str((_FINETUNE_ROOT / f"{label}-{mode}").resolve())


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF id or local path (overrides models.yaml / env)",
    )
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset key from models.yaml (default: FINETUNE_PRESET or ACTIVE_MODEL)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HF Hub repo id (e.g. tatsu-lab/alpaca) or local file path",
    )
    p.add_argument(
        "--dataset-config",
        type=str,
        default=os.environ.get("FINETUNE_DATASET_CONFIG"),
        help="HF dataset config/subset name (optional)",
    )
    p.add_argument(
        "--dataset-split",
        type=str,
        default=os.environ.get("FINETUNE_DATASET_SPLIT", "train"),
        help="HF split name or slice, e.g. train or train[:1000]",
    )
    p.add_argument(
        "--dataset-max-samples",
        type=int,
        default=int(os.environ["FINETUNE_MAX_SAMPLES"])
        if os.environ.get("FINETUNE_MAX_SAMPLES")
        else None,
        help="Cap examples after loading (useful for Hub smoke tests)",
    )
    p.add_argument(
        "--mix-json",
        type=str,
        default=os.environ.get("FINETUNE_MIX_JSON"),
        help=(
            "JSON list of dataset source specs to mix/replay; overrides "
            "--dataset/--format. Each spec: "
            '{"dataset":..,"format":..,"columns":{..},"dataset_config":..,'
            '"dataset_split":..,"max_samples":..,"max_len":..,"weight":..}'
        ),
    )
    p.add_argument(
        "--format",
        type=str,
        default=os.environ.get("FINETUNE_FORMAT", "chat"),
        choices=["alpaca", "chat", "prompt", "text"],
    )
    # Column-name overrides: let a dataset's own columns map onto a --format
    # without preprocessing (e.g. MetaMathQA query/response -> prompt format,
    # orca-math question/answer -> prompt format).
    p.add_argument("--prompt-key", default=None,
                   help="column to use as the prompt (prompt format)")
    p.add_argument("--response-key", default=None,
                   help="column to use as the response (prompt format)")
    p.add_argument("--instruction-key", default=None,
                   help="column to use as instruction (alpaca format)")
    p.add_argument("--input-key", default=None,
                   help="column to use as optional input (alpaca format)")
    p.add_argument("--output-key", default=None,
                   help="column to use as output (alpaca format)")
    p.add_argument("--mode", type=str, default="lora",
                   choices=["full", "lora", "qlora"])
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: ./models/finetuned/<preset>-<mode>)",
    )
    # training hparams
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 2e-4 for (q)lora, 2e-5 for full")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--mask_prompt", action="store_true", default=True,
                   help="compute loss only on the response tokens")
    p.add_argument("--no_mask_prompt", dest="mask_prompt", action="store_false")
    # training schedule / regularization (previously hardcoded)
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   help="LR scheduler type: cosine, linear, constant, ...")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=None,
                   help="eval every N steps (default: max_steps//5, else 200)")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--early_stopping_patience", type=int, default=0,
                   help=">0 enables early stopping + load_best_model_at_end on eval_loss")
    p.add_argument("--neftune_noise_alpha", type=float, default=None,
                   help="NEFTune noise alpha (e.g. 5) — quick instruction-tuning gain")
    p.add_argument("--report_to", type=str, default="none",
                   help="trainer reporting: none, wandb, tensorboard, ...")
    # lora hparams
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="comma list; 'all-linear' also works")
    # misc
    p.add_argument("--val_split", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=None)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("FINETUNE_DEVICE", "auto"),
        choices=["auto", "cpu", "cuda"],
        help="Training device (default: auto; set FINETUNE_DEVICE=cpu to avoid GPU OOM)",
    )
    p.add_argument("--resume", type=str, default=None)
    # merge mode
    p.add_argument("--merge", type=str, default=None,
                   help="path to a LoRA adapter dir to merge into its base model")
    p.add_argument(
        "--lm-eval-after",
        action="store_true",
        help="run slm-lm-eval on the saved checkpoint after training",
    )
    p.add_argument(
        "--lm-eval-config",
        type=str,
        default=str(_REPO_ROOT / "research/evals/configs/lm_eval_smoke.yaml"),
        help="YAML config for post-training lm-eval (default: lm_eval_smoke.yaml)",
    )
    p.add_argument(
        "--lm-eval-baseline",
        type=str,
        default=None,
        help="optional baseline preset key; runs lm-eval on base model and compares",
    )
    return p.parse_args()


# ----------------------------------------------------------------------------
# Dataset loading + normalization to (prompt, response) or raw text
# ----------------------------------------------------------------------------
def load_raw_dataset(
    path: str,
    *,
    config: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
):
    """Load from a local file or Hugging Face Hub (datasets.load_dataset)."""
    if os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".json", ".jsonl"):
            ds = load_dataset("json", data_files=path, split="train")
        elif ext == ".csv":
            ds = load_dataset("csv", data_files=path, split="train")
        elif ext == ".txt":
            ds = load_dataset("text", data_files=path, split="train")
        else:
            raise ValueError(f"Unsupported local file type: {ext}")
    else:
        kwargs: dict = {"path": path, "split": split}
        if config:
            kwargs["name"] = config
        print(f"Loading Hub dataset: {path}" + (f" (config={config})" if config else "")
              + f" split={split}")
        ds = load_dataset(**kwargs)

    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _last_metric(history: list[dict], key: str) -> float | None:
    for entry in reversed(history):
        if key in entry:
            return float(entry[key])
    return None


def _result_score(eval_loss: float | None, train_loss: float | None) -> float | None:
    """Higher is better (0–100). Derived from eval loss, else train loss."""
    loss = eval_loss if eval_loss is not None else train_loss
    if loss is None:
        return None
    # exp(-loss) maps typical LM losses (~0.5–3) into a readable 0–100 band.
    return round(min(100.0, max(0.0, 100.0 * math.exp(-loss))), 2)


def save_training_results(
    out_dir: str,
    *,
    args,
    preset_key: str | None,
    train_count: int,
    eval_count: int,
    train_result,
    log_history: list[dict],
    eval_metrics: dict | None,
) -> Path:
    history = train_result.metrics if hasattr(train_result, "metrics") else {}

    final_train_loss = _last_metric(log_history, "loss")
    if final_train_loss is None and "train_loss" in history:
        final_train_loss = float(history["train_loss"])

    eval_loss = None
    perplexity = None
    if eval_metrics:
        eval_loss = float(eval_metrics.get("eval_loss", 0))
        if eval_loss < 20:
            perplexity = round(math.exp(eval_loss), 4)

    result_score = _result_score(eval_loss, final_train_loss)

    payload = {
        "model": args.model,
        "preset": preset_key,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "mix": json.loads(args.mix_json) if args.mix_json else None,
        "format": args.format,
        "mode": args.mode,
        "output_dir": out_dir,
        "samples": {"train": train_count, "eval": eval_count},
        "metrics": {
            "final_train_loss": round(final_train_loss, 6)
            if final_train_loss is not None
            else None,
            "eval_loss": round(eval_loss, 6) if eval_loss is not None else None,
            "perplexity": perplexity,
            "loss_score": round(eval_loss, 6)
            if eval_loss is not None
            else (
                round(final_train_loss, 6) if final_train_loss is not None else None
            ),
            "result_score": result_score,
        },
        "training": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "global_step": getattr(train_result, "global_step", None),
            "train_runtime_sec": round(history.get("train_runtime", 0), 2)
            if history
            else None,
            "train_samples_per_second": history.get("train_samples_per_second"),
        },
    }

    path = Path(out_dir) / "training_results.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def to_prompt_response(example, fmt, tokenizer, keys=None):
    """Normalize any supported format into a single training string,
    returning (full_text, prompt_text). prompt_text is None for raw text.

    `keys` optionally remaps a dataset's column names onto the format's
    expected fields (e.g. {"prompt": "query"} for MetaMathQA)."""
    keys = keys or {}
    if fmt == "text":
        return example[keys.get("text", "text")], None

    if fmt == "alpaca":
        instr = example.get(keys.get("instruction", "instruction"), "")
        inp = example.get(keys.get("input", "input"), "") or ""
        out = example.get(keys.get("output", "output"), "")
        user = instr if not inp else f"{instr}\n\n{inp}"
        messages = [{"role": "user", "content": user},
                    {"role": "assistant", "content": out}]

    elif fmt == "prompt":
        prompt = example.get(keys.get("prompt", "prompt"), "")
        rkey = keys.get("response")
        resp = example.get(rkey, "") if rkey else example.get(
            "completion", example.get("response", ""))
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp}]

    elif fmt == "chat":
        messages = example["messages"]

    else:
        raise ValueError(fmt)

    # Use the model's chat template when it has one; else simple fallback.
    if tokenizer.chat_template:
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_only = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)
    else:
        prompt_only = "".join(
            f"### {m['role'].capitalize()}:\n{m['content']}\n\n"
            for m in messages[:-1]) + "### Assistant:\n"
        full = prompt_only + messages[-1]["content"] + (tokenizer.eos_token or "")
    return full, prompt_only


def build_tokenize_fn(tokenizer, fmt, max_len, mask_prompt, keys=None):
    def fn(example):
        full, prompt = to_prompt_response(example, fmt, tokenizer, keys)
        ids = tokenizer(full, truncation=True, max_length=max_len,
                        add_special_tokens=(fmt == "text"))["input_ids"]
        labels = list(ids)
        if mask_prompt and prompt is not None:
            p_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            p_len = min(p_len, len(labels))
            labels[:p_len] = [IGNORE_INDEX] * p_len     # no loss on prompt
        return {"input_ids": ids, "labels": labels}
    return fn


def _source_specs(args) -> list[dict]:
    """Return the list of dataset source specs to train on.

    With --mix-json, parse the JSON list verbatim. Otherwise synthesize a
    single source from the top-level --dataset/--format/--*-key args."""
    if args.mix_json:
        specs = json.loads(args.mix_json)
        if not isinstance(specs, list) or not specs:
            raise SystemExit("--mix-json must be a non-empty JSON list of source specs")
        return specs
    return [{
        "dataset": args.dataset,
        "format": args.format,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "max_samples": args.dataset_max_samples,
        "columns": {k: v for k, v in {
            "prompt": args.prompt_key, "response": args.response_key,
            "instruction": args.instruction_key, "input": args.input_key,
            "output": args.output_key,
        }.items() if v},
    }]


def _apply_weight(ds, weight):
    """Up-sample (weight > 1, with repeats) or sub-sample (weight < 1) a source."""
    if not weight or weight == 1.0 or len(ds) == 0:
        return ds
    target = max(0, int(round(len(ds) * float(weight))))
    if target == 0:
        return ds.select([])
    n = len(ds)
    return ds.select([i % n for i in range(target)])  # repeats when target > n


def build_training_dataset(args, tokenizer):
    """Load, tokenize, weight and concatenate every source into one dataset.

    Each source carries its own format / columns / split / max_len so a skill
    dataset can be mixed with a general-data replay slice in one run."""
    from datasets import concatenate_datasets

    specs = _source_specs(args)
    multi = len(specs) > 1
    if multi:
        print(f"Mixing {len(specs)} dataset source(s):")

    parts = []
    for i, spec in enumerate(specs):
        dataset = spec.get("dataset")
        if not dataset:
            raise SystemExit(f"mix source #{i} is missing 'dataset'")
        fmt = spec.get("format", args.format)
        raw = load_raw_dataset(
            dataset,
            config=spec.get("dataset_config"),
            split=spec.get("dataset_split", "train"),
            max_samples=spec.get("max_samples"),
        )
        raw = raw.shuffle(seed=args.seed)
        keys = spec.get("columns") or {}
        max_len = spec.get("max_len", args.max_len)
        tokenize = build_tokenize_fn(tokenizer, fmt, max_len, args.mask_prompt, keys)
        tok = raw.map(tokenize, remove_columns=raw.column_names,
                      desc=f"tokenizing {dataset}")
        tok = tok.filter(lambda e: len(e["input_ids"]) > 1)
        tok = _apply_weight(tok, spec.get("weight"))
        if multi:
            wnote = f" (weight {spec['weight']})" if spec.get("weight") else ""
            print(f"  - {dataset} [{fmt}] -> {len(tok)} examples{wnote}")
        parts.append(tok)

    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return ds.shuffle(seed=args.seed)


class CausalCollator:
    """Pads input_ids with pad_token and labels with IGNORE_INDEX."""
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        pad = self.tok.pad_token_id
        for b in batch:
            n = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * n)
            labels.append(b["labels"] + [IGNORE_INDEX] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


# ----------------------------------------------------------------------------
# Model loading for each mode
# ----------------------------------------------------------------------------
def _training_uses_cuda(args) -> bool:
    if args.device == "cpu":
        return False
    if args.device == "cuda":
        return True
    return torch.cuda.is_available()


def _gpu_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return (
        f"{free // 2**20} MiB free / {total // 2**20} MiB total "
        f"(allocated {alloc // 2**20} MiB, reserved {reserved // 2**20} MiB)"
    )


def _gpu_total_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    _, total = torch.cuda.mem_get_info()
    return total / (1024**3)


def _apply_low_vram_defaults(args) -> None:
    """Cap batch/seq length and prefer QLoRA on GPUs that cannot fit full LoRA."""
    if not _training_uses_cuda(args):
        return
    total_gib = _gpu_total_gib()
    if total_gib is None or total_gib >= 6.0:
        return

    orig_batch, orig_max_len, orig_mode = args.batch_size, args.max_len, args.mode
    args.batch_size = min(args.batch_size, 1)
    args.max_len = min(args.max_len, 512)
    args.gradient_checkpointing = True

    if total_gib < 4.5 and args.mode == "lora":
        try:
            import bitsandbytes  # noqa: F401
            args.mode = "qlora"
        except ImportError:
            print(
                f"Warning: {total_gib:.1f} GiB GPU — full LoRA may OOM. "
                "Install finetune extras and use --mode qlora:\n"
                "  uv sync --group finetune"
            )

    if (
        args.batch_size != orig_batch
        or args.max_len != orig_max_len
        or args.mode != orig_mode
    ):
        print(
            f"Low VRAM ({total_gib:.1f} GiB): adjusted training defaults — "
            f"batch_size {orig_batch}->{args.batch_size}, "
            f"max_len {orig_max_len}->{args.max_len}"
            + (f", mode {orig_mode}->{args.mode}" if args.mode != orig_mode else "")
        )


def _validate_cuda_device(args) -> None:
    if not _training_uses_cuda(args):
        return
    if torch.cuda.is_available():
        return
    raise SystemExit(
        "CUDA training was requested (--device cuda or auto with a visible GPU) "
        "but PyTorch cannot use the GPU.\n"
        f"  torch.cuda.is_available() = False\n"
        f"  torch.cuda.device_count() = {torch.cuda.device_count()}\n"
        "Run `nvidia-smi` and check for driver errors (ERR! fields). "
        "If the GPU is busy or broken, free it or reboot, then retry.\n"
        "Fallback: pass --device cpu (slower, higher RAM use)."
    )


def clear_gpu_memory(*, reset_peak: bool = True) -> None:
    """Release cached CUDA allocations before loading a model."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _cuda_device_map() -> str | dict[str, int]:
    """Keep weights on one GPU; avoid CPU offload on small cards."""
    if torch.cuda.device_count() <= 1:
        return {"": 0}
    return "auto"


def load_model_and_tokenizer(args):
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = _training_uses_cuda(args)
    bf16_ok = (
        args.bf16
        if args.bf16 is not None
        else use_cuda and torch.cuda.is_bf16_supported()
    )
    dtype = torch.bfloat16 if bf16_ok else torch.float32

    if args.mode == "qlora":
        if not use_cuda:
            raise SystemExit("QLoRA requires CUDA. Use --mode lora with --device cpu.")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "QLoRA requires bitsandbytes. Install with:\n"
                "  uv sync --group finetune"
            ) from exc
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            device_map=_cuda_device_map(),
            **common,
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=_cuda_device_map() if use_cuda else None,
            **common,
        )
        if not use_cuda:
            model.to("cpu")

    if args.mode in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model
        targets = ("all-linear" if args.lora_targets == "all-linear"
                   else [t.strip() for t in args.lora_targets.split(",")])
        cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer, bf16_ok


# ----------------------------------------------------------------------------
# Merge a trained LoRA adapter back into base weights
# ----------------------------------------------------------------------------
def merge_adapter(adapter_dir, out_dir):
    from peft import PeftModel, PeftConfig
    cfg = PeftConfig.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")


def run_post_lm_eval(
    *,
    checkpoint_path: str,
    config_path: str,
    experiment_name: str,
    baseline_preset: str | None = None,
    adapter_path: str | None = None,
) -> dict | None:
    """Run slm-lm-eval via subprocess; return paths written under post_eval."""
    baseline_results: Path | None = None
    if baseline_preset:
        baseline_name = f"{baseline_preset}__lm-eval-baseline"
        baseline_cmd = [
            "uv",
            "run",
            "--package",
            "slm-evals",
            "slm-lm-eval",
            "--config",
            config_path,
            "--preset",
            baseline_preset,
            "--experiment-name",
            baseline_name,
        ]
        print(f"\n--- lm-eval baseline ({baseline_preset}) ---")
        subprocess.run(baseline_cmd, cwd=_REPO_ROOT, check=False)
        baseline_results = (
            _REPO_ROOT / "results" / "lm_eval" / baseline_name / "results.json"
        )

    cmd = [
        "uv",
        "run",
        "--package",
        "slm-evals",
        "slm-lm-eval",
        "--config",
        config_path,
        "--model",
        checkpoint_path,
        "--experiment-name",
        experiment_name,
    ]
    if adapter_path:
        cmd.extend(["--adapter", adapter_path])
    if baseline_results and baseline_results.is_file():
        cmd.extend(["--compare-to", str(baseline_results)])

    print(f"\n--- lm-eval candidate ({experiment_name}) ---")
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, check=False)
    out_root = _REPO_ROOT / "results" / "lm_eval" / experiment_name
    post_eval = {
        "experiment_name": experiment_name,
        "config": config_path,
        "checkpoint_path": checkpoint_path,
        "adapter_path": adapter_path,
        "baseline_preset": baseline_preset,
        "results_json": str(out_root / "results.json"),
        "summary_md": str(out_root / "summary.md"),
        "comparison_md": str(out_root / "comparison.md")
        if (out_root / "comparison.md").is_file()
        else None,
        "exit_code": proc.returncode,
    }
    return post_eval if proc.returncode == 0 else post_eval


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    _load_dotenv(_REPO_ROOT / ".env")
    args = parse_args()

    if args.merge:
        out_dir = args.out or default_output_dir(None, "merged")
        merge_adapter(args.merge, out_dir)
        return

    model_id, preset_key, trust_remote_code = resolve_model_and_preset(
        model_arg=args.model,
        preset_arg=args.preset,
    )
    args.model = model_id
    args.trust_remote_code = trust_remote_code
    if not args.dataset:
        args.dataset = (
            os.environ.get("FINETUNE_DATASET")
            or str(_DEFAULT_DATASET)
        )
    if not args.out:
        args.out = os.environ.get("FINETUNE_OUT") or default_output_dir(
            preset_key, args.mode
        )

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print(f"Base model: {args.model}")
    if preset_key:
        print(f"Preset: {preset_key}")
    if args.mix_json:
        print(f"Dataset mix: {len(json.loads(args.mix_json))} source(s)")
    else:
        print(f"Dataset: {args.dataset}")
    print(f"Output: {args.out}")
    print(f"Device: {args.device}")

    _validate_cuda_device(args)
    _apply_low_vram_defaults(args)
    if _training_uses_cuda(args):
        print(f"GPU before cleanup: {_gpu_memory_summary()}")
        clear_gpu_memory()
        print(f"GPU after cleanup:  {_gpu_memory_summary()}")

    lr = args.lr or (2e-5 if args.mode == "full" else 2e-4)

    model, tokenizer, bf16_ok = load_model_and_tokenizer(args)

    if _training_uses_cuda(args):
        print(f"GPU after model load: {_gpu_memory_summary()}")

    ds = build_training_dataset(args, tokenizer)

    if args.val_split > 0:
        split = ds.train_test_split(test_size=args.val_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    # Default eval cadence to the run length so short (max_steps) runs still
    # evaluate mid-training instead of only at the end.
    eval_steps = args.eval_steps
    if eval_steps is None:
        eval_steps = max(1, args.max_steps // 5) if args.max_steps > 0 else 200

    use_best = args.early_stopping_patience > 0 and eval_ds is not None
    # load_best_model_at_end needs save_steps aligned to eval_steps.
    save_steps = eval_steps if use_best else args.save_steps

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=use_best,
        metric_for_best_model="eval_loss" if use_best else None,
        greater_is_better=False if use_best else None,
        bf16=bf16_ok,
        fp16=(not bf16_ok and _training_uses_cuda(args)),
        gradient_checkpointing=args.gradient_checkpointing,
        neftune_noise_alpha=args.neftune_noise_alpha,
        report_to=args.report_to,
        seed=args.seed,
    )

    callbacks = []
    if use_best:
        from transformers import EarlyStoppingCallback
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalCollator(tokenizer),
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume)

    # ---- save -----------------------------------------------------------
    model.config.use_cache = True
    trainer.save_model(args.out)            # full weights OR adapter only
    tokenizer.save_pretrained(args.out)

    eval_metrics = None
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        ppl = (
            math.exp(eval_metrics["eval_loss"])
            if eval_metrics["eval_loss"] < 20
            else float("inf")
        )
        print(
            f"\neval_loss={eval_metrics['eval_loss']:.4f}  "
            f"perplexity={ppl:.2f}"
        )

    results_path = save_training_results(
        args.out,
        args=args,
        preset_key=preset_key,
        train_count=len(train_ds),
        eval_count=len(eval_ds) if eval_ds is not None else 0,
        train_result=train_result,
        log_history=trainer.state.log_history,
        eval_metrics=eval_metrics,
    )
    m = json.loads(results_path.read_text())["metrics"]
    print("\n--- scores ---")
    print(f"loss_score   = {m['loss_score']}  (lower is better)")
    print(f"result_score = {m['result_score']}  (0–100, higher is better)")
    print(f"Saved to {results_path}")

    if args.mode in ("lora", "qlora"):
        merged = f"{args.out}-merged"
        print(f"\nAdapter saved to {args.out}")
        print(
            "Use in Gradio: set ACTIVE_MODEL to the matching *-lora preset "
            "in models.yaml, or merge with:\n"
            f"  python research/finetune.py --merge {args.out} --out {merged}"
        )
    else:
        print(f"\nFull model saved to {args.out}")

    # quick smoke generation
    try:
        model.eval()
        prompt = "Hello! Briefly introduce yourself."
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        device = next(model.parameters()).device
        ids = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=60, do_sample=True,
                             temperature=0.7,
                             pad_token_id=tokenizer.pad_token_id)
        print("\n--- sample ---\n" +
              tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True))
    except Exception as e:                  # smoke test is best-effort
        print(f"(sample generation skipped: {e})")

    if args.lm_eval_after:
        exp_name = f"{Path(args.out).name}__lm-eval-posttrain"
        if args.mode in ("lora", "qlora"):
            post_eval = run_post_lm_eval(
                checkpoint_path=args.model,
                config_path=args.lm_eval_config,
                experiment_name=exp_name,
                baseline_preset=args.lm_eval_baseline or preset_key,
                adapter_path=args.out,
            )
        else:
            post_eval = run_post_lm_eval(
                checkpoint_path=args.out,
                config_path=args.lm_eval_config,
                experiment_name=exp_name,
                baseline_preset=args.lm_eval_baseline or preset_key,
            )
        if post_eval:
            payload = json.loads(results_path.read_text())
            payload["post_eval"] = post_eval
            results_path.write_text(json.dumps(payload, indent=2))
            print(f"Appended post_eval to {results_path}")

    if _training_uses_cuda(args):
        clear_gpu_memory()
        print(f"GPU after training: {_gpu_memory_summary()}")


if __name__ == "__main__":
    main()
