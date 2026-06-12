"""Joint pretrain: LLM (LoRA) + embedder + JEPA + bridge, saved to models/ensemble/."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from ensemble.checkpoint import save_checkpoint
from ensemble.jepa_ensemble import Ensemble, segment_pairs_from_texts

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATA = _REPO_ROOT / "research/data/education-lesson-chat.jsonl"
_DEFAULT_KB = _REPO_ROOT / "research/data/benchmark-kb.jsonl"
_DEFAULT_OUT = _REPO_ROOT / "models/ensemble/jepa-lesson-pretrain"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _chat_to_text(row: dict) -> str:
    messages = row.get("messages", [])
    parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    return "\n".join(parts)


def _collect_texts(data_path: Path, max_samples: int | None) -> list[str]:
    rows = _load_jsonl(data_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    return [_chat_to_text(r) for r in rows if _chat_to_text(r).strip()]


def _seed_memory(ens: Ensemble, kb_path: Path | None) -> int:
    if kb_path is None or not kb_path.is_file():
        return 0
    count = 0
    for row in _load_jsonl(kb_path):
        text = row.get("text", "").strip()
        if text:
            ens.memorize_text(text)
            count += 1
    return count


def pretrain(args) -> Path:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_path = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    kb_path = Path(args.kb).resolve() if args.kb else None

    print(f"Loading ensemble backend: {args.llm}")
    ens = Ensemble(llm=args.llm, load_in_4bit=args.load_in_4bit)
    opt = ens.make_optimizer(lr_lora=args.lr_lora, lr_aux=args.lr_aux)

    texts = _collect_texts(data_path, args.max_samples)
    if not texts and args.llm != "tiny":
        raise SystemExit(f"No training texts found in {data_path}")

    mem_count = _seed_memory(ens, kb_path)
    print(f"Training texts: {len(texts)} | memory snippets: {mem_count}")

    if args.llm == "tiny":
        n_pairs = max(args.steps * args.batch_size, args.batch_size)
        v = ens.llm.vocab_size
        seg_a = torch.randint(0, v, (n_pairs, args.seg_len))
        seg_b = torch.randint(0, v, (n_pairs, args.seg_len))
    else:
        seg_a, seg_b = segment_pairs_from_texts(
            ens.llm, texts, seg_len=args.seg_len
        )
    n_pairs = seg_a.size(0)
    batch = min(args.batch_size, n_pairs)
    print(f"Segment pairs: {n_pairs} | batch={batch} | steps={args.steps}")

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, n_pairs, (batch,))
        logs = ens.train_step(seg_a[idx], seg_b[idx], opt, w_bridge=args.w_bridge)
        if step % max(1, args.log_every) == 0 or step == args.steps - 1:
            parts = " | ".join(f"{k} {v:.4f}" for k, v in logs.items())
            print(f"step {step:4d}/{args.steps} | {parts}")

    elapsed = time.time() - t0
    meta = {
        "steps": args.steps,
        "batch_size": batch,
        "seg_len": args.seg_len,
        "data": str(data_path),
        "kb": str(kb_path) if kb_path else None,
        "memory_count": mem_count,
        "text_count": len(texts),
        "elapsed_s": round(elapsed, 1),
        "lr_lora": args.lr_lora,
        "lr_aux": args.lr_aux,
        "w_bridge": args.w_bridge,
        "seed": args.seed,
    }

    saved = save_checkpoint(
        ens,
        out_dir,
        base_llm=args.llm,
        training_meta=meta,
    )
    print(f"\nSaved ensemble checkpoint → {saved}")
    print("Benchmark with slm-evals:")
    print(
        f"  uv run --package slm-evals slm-benchmark "
        f"--model {saved} --model-type ensemble "
        f"--benchmarks bfcl --max-samples 5"
    )
    return saved


def parse_args():
    p = argparse.ArgumentParser(
        description="Pretrain JEPA ensemble (LLM+emb+JEPA) and save to models/ensemble/"
    )
    p.add_argument(
        "--llm",
        default="tiny",
        help="'tiny' for CPU smoke, or HF hub id / local path",
    )
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Chat JSONL (messages[]) for segment-pair training",
    )
    p.add_argument(
        "--kb",
        default=str(_DEFAULT_KB),
        help="Optional KB JSONL (text field) loaded into vector store",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Output directory (default: models/ensemble/jepa-lesson-pretrain)",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seg-len", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-aux", type=float, default=1e-3)
    p.add_argument("--w-bridge", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--no-kb", action="store_true", help="Skip loading KB into memory")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_kb:
        args.kb = None
    pretrain(args)


if __name__ == "__main__":
    main()
