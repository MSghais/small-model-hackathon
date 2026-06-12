"""
eval_harness.py — Ablation ladder + JEPA best-of-N test for the ensemble
========================================================================
Companion to `llm_emb_jepa_ensemble_pluggable.py` (must be importable,
i.e. in the same directory).

What it runs
------------
1. ABLATION LADDER on a QA set:
     C1  base LLM alone
     C2  C1 + RAG (embedding retrieval)
     C3  C2 + router/adapters
     C4  C3 + JEPA best-of-N critic
   (C5 = C4 with a bridge-trained checkpoint — just pass --ckpt)

2. BEST-OF-N SELECTOR comparison (the decisive JEPA experiment):
     first-sample | random-pick | JEPA-score pick | oracle pick
   All on the SAME N drafts per question, so differences are pure selection.

3. CONTINUAL FORGETTING test (optional, --continual):
     accuracy on task A before vs after training adapters for B and C.

4. PAIRED BOOTSTRAP significance between any two configs.

Usage
-----
# Smoke test, no GPU/deps beyond torch (toy backend, synthetic QA):
python eval_harness.py --llm tiny --toy

# Real model + your QA file (jsonl: {"question": ..., "answer": ..., "context": optional}):
python eval_harness.py --llm Qwen/Qwen2.5-0.5B-Instruct \
    --qa ./domain_qa.jsonl --kb ./knowledge.jsonl --n_drafts 8

# With a bridge-trained ensemble checkpoint (C5):
python eval_harness.py --llm /models/llama-3.2-1b --qa ./qa.jsonl \
    --kb ./kb.jsonl --ckpt ./ensemble_bridge.pt

QA file:  {"question": str, "answer": str, "domain": optional str}
KB file:  {"text": str}   (each line becomes one memory in the vector store)
"""

import argparse
import json
import random
import re
import string
import time
from collections import Counter, defaultdict

import torch

from llm_emb_jepa_ensemble_pluggable import Ensemble  # same directory

# ----------------------------------------------------------------------------
# Metrics: normalized exact match + token F1 (SQuAD-style)
# ----------------------------------------------------------------------------
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(pred: str, gold: str) -> float:
    return float(normalize(gold) in normalize(pred))   # containment EM


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


# ----------------------------------------------------------------------------
# Paired bootstrap: P(config B beats config A)
# ----------------------------------------------------------------------------
def paired_bootstrap(scores_a, scores_b, iters=2000, seed=0):
    rng = random.Random(seed)
    n, wins = len(scores_a), 0
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(scores_a[i] for i in idx) / n
        db = sum(scores_b[i] for i in idx) / n
        wins += db > da
    return wins / iters


# ----------------------------------------------------------------------------
# Config runners — each returns per-question dicts
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_plain(ens, q_ids, n_new):
    """C1: base adapter, no retrieval, single sample."""
    ens.llm.set_adapter(ens.adapter_names[0])
    t0 = time.time()
    out = ens.llm.generate(q_ids.to(ens.llm.device), n_new=n_new, temperature=0.7)
    return out[:, q_ids.size(1):], time.time() - t0


@torch.no_grad()
def generate_config(ens, q_ids, n_new, *, use_rag, use_router, use_jepa,
                    n_drafts=1, tau=0.0):
    """Unified runner for C2/C3/C4."""
    q_emb = ens.emb(q_ids.cpu())

    if use_router:
        a_idx = ens.router(q_emb).item()
        ens.llm.set_adapter(ens.adapter_names[a_idx])
    else:
        ens.llm.set_adapter(ens.adapter_names[0])

    ctx = q_ids.cpu()
    if use_rag:
        mems = ens.store.search(q_emb, k=1)
        if mems:
            ctx = torch.cat([mems[0], ctx], dim=1)

    t0 = time.time()
    if not use_jepa:
        out = ens.llm.generate(ctx.to(ens.llm.device), n_new=n_new, temperature=0.7)
        return out[:, ctx.size(1):], time.time() - t0, None

    # JEPA best-of-N: sample drafts, keep the one closest to predicted latent
    z_exp = ens.jepa.predict_next_latent(ctx)
    drafts, scores = [], []
    for _ in range(n_drafts):
        out = ens.llm.generate(ctx.to(ens.llm.device), n_new=n_new, temperature=0.9)
        new = out[:, ctx.size(1):].cpu()
        drafts.append(new)
        scores.append(torch.nn.functional.cosine_similarity(
            z_exp, ens.jepa.encode(new)).item())
    best = max(range(n_drafts), key=lambda i: scores[i])
    return drafts[best], time.time() - t0, (drafts, scores)


# ----------------------------------------------------------------------------
# Best-of-N selector comparison on shared drafts
# ----------------------------------------------------------------------------
def selector_comparison(drafts_scores_gold, decode_fn, rng):
    """drafts_scores_gold: list of (drafts, jepa_scores, gold_answer).
    Returns EM for: first | random | jepa | oracle — all on the SAME drafts."""
    res = defaultdict(list)
    for drafts, scores, gold in drafts_scores_gold:
        texts = [decode_fn(d) for d in drafts]
        ems = [em_score(t, gold) for t in texts]
        res["first"].append(ems[0])
        res["random"].append(ems[rng.randrange(len(ems))])
        res["jepa"].append(ems[max(range(len(ems)), key=lambda i: scores[i])])
        res["oracle"].append(max(ems))     # upper bound of selection
    return {k: sum(v) / len(v) for k, v in res.items()}, res


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_toy_data(ens, n_qa=20, vocab=None):
    """Synthetic QA for the tiny backend: 'answer' token sequence is planted
    in the KB so RAG can genuinely help even with random weights."""
    vocab = vocab or ens.llm.vocab_size
    qa, kb = [], []
    for i in range(n_qa):
        key = torch.randint(0, vocab, (1, 6))
        ans = torch.randint(0, vocab, (1, 4))
        kb.append(torch.cat([key, ans], dim=1))            # memory = key+answer
        qa.append({"q_ids": key, "answer_ids": ans})
    return qa, kb


# ----------------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------------
def run(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    ens = Ensemble(llm=args.llm)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        ens.load_state_dict(state, strict=False)
        print(f"loaded ensemble checkpoint: {args.ckpt}")

    is_text = args.llm != "tiny"

    # ---- load data and fill the vector store -------------------------------
    if args.toy or not is_text:
        qa, kb = make_toy_data(ens)
        for mem in kb:
            ens.memorize_ids(mem)
        def to_ids(item):  return item["q_ids"]
        def gold_of(item): return item["answer_ids"]
        def decode(ids):   return " ".join(map(str, ids[0].tolist()))
        def gold_text(item): return decode(item["answer_ids"])
    else:
        qa = load_jsonl(args.qa)
        if args.kb:
            for row in load_jsonl(args.kb):
                ens.memorize_text(row["text"])
        def to_ids(item):  return ens.llm.encode_text(
            f"Answer briefly.\nQ: {item['question']}\nA:")
        def gold_text(item): return item["answer"]
        def decode(ids):   return ens.llm.decode(ids)

    qa = qa[: args.limit]
    print(f"eval set: {len(qa)} questions | store: {len(ens.store.keys)} memories\n")

    # ---- ablation ladder ----------------------------------------------------
    configs = {
        "C1_base":        dict(use_rag=False, use_router=False, use_jepa=False),
        "C2_rag":         dict(use_rag=True,  use_router=False, use_jepa=False),
        "C3_rag_router":  dict(use_rag=True,  use_router=True,  use_jepa=False),
        "C4_full_jepa":   dict(use_rag=True,  use_router=True,  use_jepa=True,
                               n_drafts=args.n_drafts),
    }

    per_q = {}            # config -> list of EM scores (for bootstrap)
    summary = {}
    jepa_material = []    # (drafts, scores, gold) for selector comparison

    for name, cfg in configs.items():
        ems, f1s, lats = [], [], []
        for item in qa:
            ids = to_ids(item)
            if name == "C1_base":
                out, dt = generate_plain(ens, ids, args.n_new)
                extra = None
            else:
                out, dt, extra = generate_config(ens, ids, args.n_new, **cfg)
            pred, gold = decode(out), gold_text(item)
            ems.append(em_score(pred, gold))
            f1s.append(f1_score(pred, gold))
            lats.append(dt)
            if name == "C4_full_jepa" and extra is not None:
                jepa_material.append((extra[0], extra[1], gold))
        per_q[name] = ems
        summary[name] = (sum(ems) / len(ems), sum(f1s) / len(f1s),
                         sum(lats) / len(lats))

    print(f"{'config':<16}{'EM':>8}{'F1':>8}{'lat(s)':>9}")
    for k, (em, f1, lat) in summary.items():
        print(f"{k:<16}{em:>8.3f}{f1:>8.3f}{lat:>9.3f}")

    # deltas + significance
    print("\ncomponent contributions (paired bootstrap, P(B>A)):")
    ladder = list(configs.keys())
    for a, b in zip(ladder, ladder[1:]):
        d = summary[b][0] - summary[a][0]
        p = paired_bootstrap(per_q[a], per_q[b])
        print(f"  {b} - {a}: ΔEM={d:+.3f}   P(better)={p:.2f}")

    # ---- decisive JEPA selector experiment ----------------------------------
    if jepa_material:
        sel, sel_per_q = selector_comparison(jepa_material, decode, rng)
        print("\nbest-of-N selector comparison (same drafts, N="
              f"{args.n_drafts}):")
        for k in ("first", "random", "jepa", "oracle"):
            print(f"  {k:<8}EM={sel[k]:.3f}")
        p = paired_bootstrap(sel_per_q["random"], sel_per_q["jepa"])
        print(f"  P(jepa > random) = {p:.2f}   "
              f"{'JEPA critic WORKS' if p > 0.95 else 'inconclusive — critic ~ random'}")
        gap = sel["oracle"] - sel["jepa"]
        print(f"  headroom to oracle: {gap:.3f}")

    # ---- continual forgetting (optional) ------------------------------------
    if args.continual:
        print("\ncontinual test: accuracy on task-A questions "
              "before vs after adding adapters B and C")
        ems_before = per_q["C3_rag_router"]
        ens.new_task_adapter("task_B")
        ens.new_task_adapter("task_C")
        ems_after = []
        for item in qa:
            out, _, _ = generate_config(ens, to_ids(item), args.n_new,
                                        use_rag=True, use_router=True,
                                        use_jepa=False)
            ems_after.append(em_score(decode(out), gold_text(item)))
        bt = sum(ems_after) / len(ems_after) - sum(ems_before) / len(ems_before)
        print(f"  backward transfer (≈0 is ideal): {bt:+.3f}")

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="tiny", help="'tiny' | HF id | local path")
    p.add_argument("--qa", default=None, help="jsonl with question/answer")
    p.add_argument("--kb", default=None, help="jsonl with text -> vector store")
    p.add_argument("--ckpt", default=None, help="bridge-trained ensemble .pt (C5)")
    p.add_argument("--toy", action="store_true", help="synthetic data smoke test")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--n_new", type=int, default=24)
    p.add_argument("--n_drafts", type=int, default=8)
    p.add_argument("--continual", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
