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
"""

import argparse
import gc
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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
        help="HF dataset id or local file path",
    )
    p.add_argument(
        "--format",
        type=str,
        default=os.environ.get("FINETUNE_FORMAT", "chat"),
        choices=["alpaca", "chat", "prompt", "text"],
    )
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
    return p.parse_args()


# ----------------------------------------------------------------------------
# Dataset loading + normalization to (prompt, response) or raw text
# ----------------------------------------------------------------------------
def load_raw_dataset(path: str):
    if os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".json", ".jsonl"):
            return load_dataset("json", data_files=path, split="train")
        if ext == ".csv":
            return load_dataset("csv", data_files=path, split="train")
        if ext == ".txt":
            return load_dataset("text", data_files=path, split="train")
        raise ValueError(f"Unsupported local file type: {ext}")
    return load_dataset(path, split="train")          # Hub id


def to_prompt_response(example, fmt, tokenizer):
    """Normalize any supported format into a single training string,
    returning (full_text, prompt_text). prompt_text is None for raw text."""
    if fmt == "text":
        return example["text"], None

    if fmt == "alpaca":
        instr = example.get("instruction", "")
        inp = example.get("input", "") or ""
        out = example.get("output", "")
        user = instr if not inp else f"{instr}\n\n{inp}"
        messages = [{"role": "user", "content": user},
                    {"role": "assistant", "content": out}]

    elif fmt == "prompt":
        prompt = example.get("prompt", "")
        resp = example.get("completion", example.get("response", ""))
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


def build_tokenize_fn(tokenizer, fmt, max_len, mask_prompt):
    def fn(example):
        full, prompt = to_prompt_response(example, fmt, tokenizer)
        ids = tokenizer(full, truncation=True, max_length=max_len,
                        add_special_tokens=(fmt == "text"))["input_ids"]
        labels = list(ids)
        if mask_prompt and prompt is not None:
            p_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            p_len = min(p_len, len(labels))
            labels[:p_len] = [IGNORE_INDEX] * p_len     # no loss on prompt
        return {"input_ids": ids, "labels": labels}
    return fn


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
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.out}")
    print(f"Device: {args.device}")

    _validate_cuda_device(args)
    if _training_uses_cuda(args):
        print(f"GPU before cleanup: {_gpu_memory_summary()}")
        clear_gpu_memory()
        print(f"GPU after cleanup:  {_gpu_memory_summary()}")

    lr = args.lr or (2e-5 if args.mode == "full" else 2e-4)

    model, tokenizer, bf16_ok = load_model_and_tokenizer(args)

    if _training_uses_cuda(args):
        print(f"GPU after model load: {_gpu_memory_summary()}")

    ds = load_raw_dataset(args.dataset)
    ds = ds.shuffle(seed=args.seed)
    tokenize = build_tokenize_fn(tokenizer, args.format, args.max_len,
                                 args.mask_prompt)
    ds = ds.map(tokenize, remove_columns=ds.column_names, desc="tokenizing")
    ds = ds.filter(lambda e: len(e["input_ids"]) > 1)

    if args.val_split > 0:
        split = ds.train_test_split(test_size=args.val_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=bf16_ok,
        fp16=(not bf16_ok and _training_uses_cuda(args)),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalCollator(tokenizer),
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # ---- save -----------------------------------------------------------
    model.config.use_cache = True
    trainer.save_model(args.out)            # full weights OR adapter only
    tokenizer.save_pretrained(args.out)

    if eval_ds is not None:
        metrics = trainer.evaluate()
        ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
        print(f"\neval_loss={metrics['eval_loss']:.4f}  perplexity={ppl:.2f}")

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

    if _training_uses_cuda(args):
        clear_gpu_memory()
        print(f"GPU after training: {_gpu_memory_summary()}")


if __name__ == "__main__":
    main()
