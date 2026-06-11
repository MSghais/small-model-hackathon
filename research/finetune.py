"""
finetune.py — Fine-tune a small LLM: FULL, LoRA, or QLoRA, one script.
======================================================================

Install:
    pip install torch transformers datasets peft accelerate
    pip install bitsandbytes        # only needed for --mode qlora

Examples
--------
# LoRA on an instruction dataset from the Hub
python finetune.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset tatsu-lab/alpaca --format alpaca \
    --mode lora --epochs 1 --out ./out-lora

# QLoRA (4-bit) on a local JSONL chat file: {"messages": [{"role":..,"content":..}, ...]}
python finetune.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset ./data/chats.jsonl --format chat \
    --mode qlora --out ./out-qlora

# FULL fine-tune on raw text files (continued pretraining style)
python finetune.py \
    --model HuggingFaceTB/SmolLM2-360M \
    --dataset ./data/corpus.txt --format text \
    --mode full --lr 2e-5 --out ./out-full

# Local model path works the same way
python finetune.py --model /models/llama-3.2-1b --dataset ./data.jsonl \
    --format chat --mode lora --out ./out

# After LoRA training, optionally merge adapter into the base weights:
python finetune.py --merge ./out-lora --out ./merged-model

Dataset formats (--format)
--------------------------
  alpaca : columns instruction / input(optional) / output
  chat   : column  messages = [{"role": "...", "content": "..."}]
  prompt : columns prompt / completion  (or prompt / response)
  text   : column  text  — or a plain .txt file (one doc per line / whole file)

Local files: .json, .jsonl, .csv, .txt. Hub ids: any datasets repo.
"""

import argparse
import os
import math

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


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, help="HF id or local path of base model")
    p.add_argument("--dataset", type=str, help="HF dataset id or local file path")
    p.add_argument("--format", type=str, default="alpaca",
                   choices=["alpaca", "chat", "prompt", "text"])
    p.add_argument("--mode", type=str, default="lora",
                   choices=["full", "lora", "qlora"])
    p.add_argument("--out", type=str, default="./finetuned")
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
def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_ok = (args.bf16 if args.bf16 is not None
               else torch.cuda.is_available()
               and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16_ok else torch.float32

    if args.mode == "qlora":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto")
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None)

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
    args = parse_args()

    if args.merge:
        merge_adapter(args.merge, args.out)
        return

    assert args.model and args.dataset, "--model and --dataset are required"
    lr = args.lr or (2e-5 if args.mode == "full" else 2e-4)

    model, tokenizer, bf16_ok = load_model_and_tokenizer(args)

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
        fp16=(not bf16_ok and torch.cuda.is_available()),
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
        print(f"\nAdapter saved to {args.out}")
        print(f"Merge into standalone weights with:\n"
              f"  python finetune.py --merge {args.out} --out {args.out}-merged")
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
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=60, do_sample=True,
                             temperature=0.7,
                             pad_token_id=tokenizer.pad_token_id)
        print("\n--- sample ---\n" +
              tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True))
    except Exception as e:                  # smoke test is best-effort
        print(f"(sample generation skipped: {e})")


if __name__ == "__main__":
    main()
