"""Ablation ladder + JEPA best-of-N benchmark for the ensemble."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from ensemble.eval.metrics import em_score, f1_score, paired_bootstrap
from ensemble.backends import TinyBackend
from ensemble.checkpoint import load_checkpoint
from ensemble.config import load_dotenv, resolve_llm_cli
from ensemble.jepa_ensemble import Ensemble


@torch.no_grad()
def generate_plain(ens, q_ids, n_new):
    ens.llm.set_adapter(ens.adapter_names[0])
    t0 = time.time()
    out = ens.llm.generate(q_ids.to(ens.llm.device), n_new=n_new, temperature=0.7)
    return out[:, q_ids.size(1) :], time.time() - t0


@torch.no_grad()
def generate_config(
    ens, q_ids, n_new, *, use_rag, use_router, use_jepa, n_drafts=1, tau=0.0
):
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
        out = ens.llm.generate(
            ctx.to(ens.llm.device), n_new=n_new, temperature=0.7
        )
        return out[:, ctx.size(1) :], time.time() - t0, None

    z_exp = ens.jepa.predict_next_latent(ctx)
    drafts, scores = [], []
    for _ in range(n_drafts):
        out = ens.llm.generate(
            ctx.to(ens.llm.device), n_new=n_new, temperature=0.9
        )
        new = out[:, ctx.size(1) :].cpu()
        drafts.append(new)
        scores.append(
            F.cosine_similarity(z_exp, ens.jepa.encode(new)).item()
        )
    best = max(range(n_drafts), key=lambda i: scores[i])
    return drafts[best], time.time() - t0, (drafts, scores)


def selector_comparison(drafts_scores_gold, decode_fn, rng):
    res = defaultdict(list)
    for drafts, scores, gold in drafts_scores_gold:
        texts = [decode_fn(d) for d in drafts]
        ems = [em_score(t, gold) for t in texts]
        res["first"].append(ems[0])
        res["random"].append(ems[rng.randrange(len(ems))])
        res["jepa"].append(ems[max(range(len(ems)), key=lambda i: scores[i])])
        res["oracle"].append(max(ems))
    return {k: sum(v) / len(v) for k, v in res.items()}, res


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def make_toy_data(ens, n_qa=20, vocab=None):
    vocab = vocab or ens.llm.vocab_size
    qa, kb = [], []
    for _ in range(n_qa):
        key = torch.randint(0, vocab, (1, 6))
        ans = torch.randint(0, vocab, (1, 4))
        kb.append(torch.cat([key, ans], dim=1))
        qa.append({"q_ids": key, "answer_ids": ans})
    return qa, kb


def run(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    if args.ckpt:
        ens = load_checkpoint(args.ckpt)
        print(f"loaded ensemble checkpoint: {args.ckpt}")
        is_text = not isinstance(ens.llm, TinyBackend)
    else:
        load_dotenv()
        args.llm = resolve_llm_cli(
            args.llm, toy=args.toy, preset=getattr(args, "preset", None)
        )
        print(f"Resolved LLM: {args.llm}")
        ens = Ensemble(llm=args.llm)
        is_text = args.llm != "tiny"

    if args.toy or not is_text:
        qa, kb = make_toy_data(ens)
        for mem in kb:
            ens.memorize_ids(mem)

        def to_ids(item):
            return item["q_ids"]

        def gold_text(item):
            return " ".join(map(str, item["answer_ids"][0].tolist()))

        def decode(ids):
            return " ".join(map(str, ids[0].tolist()))
    else:
        qa = load_jsonl(args.qa)
        if args.kb:
            for row in load_jsonl(args.kb):
                ens.memorize_text(row["text"])

        def to_ids(item):
            return ens.llm.encode_text(
                f"Answer briefly.\nQ: {item['question']}\nA:"
            )

        def gold_text(item):
            return item["answer"]

        def decode(ids):
            return ens.llm.decode(ids)

    qa = qa[: args.limit]
    print(
        f"eval set: {len(qa)} questions | store: {len(ens.store.keys)} memories\n"
    )

    configs = {
        "C1_base": dict(use_rag=False, use_router=False, use_jepa=False),
        "C2_rag": dict(use_rag=True, use_router=False, use_jepa=False),
        "C3_rag_router": dict(use_rag=True, use_router=True, use_jepa=False),
        "C4_full_jepa": dict(
            use_rag=True,
            use_router=True,
            use_jepa=True,
            n_drafts=args.n_drafts,
        ),
    }

    per_q = {}
    summary = {}
    jepa_material = []

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
        summary[name] = (
            sum(ems) / len(ems),
            sum(f1s) / len(f1s),
            sum(lats) / len(lats),
        )

    print(f"{'config':<16}{'EM':>8}{'F1':>8}{'lat(s)':>9}")
    for k, (em, f1, lat) in summary.items():
        print(f"{k:<16}{em:>8.3f}{f1:>8.3f}{lat:>9.3f}")

    print("\ncomponent contributions (paired bootstrap, P(B>A)):")
    ladder = list(configs.keys())
    for a, b in zip(ladder, ladder[1:]):
        d = summary[b][0] - summary[a][0]
        p = paired_bootstrap(per_q[a], per_q[b])
        print(f"  {b} - {a}: ΔEM={d:+.3f}   P(better)={p:.2f}")

    if jepa_material:
        sel, sel_per_q = selector_comparison(jepa_material, decode, rng)
        print(
            f"\nbest-of-N selector comparison (same drafts, N={args.n_drafts}):"
        )
        for k in ("first", "random", "jepa", "oracle"):
            print(f"  {k:<8}EM={sel[k]:.3f}")
        p = paired_bootstrap(sel_per_q["random"], sel_per_q["jepa"])
        verdict = (
            "JEPA critic WORKS"
            if p > 0.95
            else "inconclusive — critic ~ random"
        )
        print(f"  P(jepa > random) = {p:.2f}   {verdict}")
        print(f"  headroom to oracle: {sel['oracle'] - sel['jepa']:.3f}")

    if args.continual:
        print(
            "\ncontinual test: accuracy on task-A questions "
            "before vs after adding adapters B and C"
        )
        ems_before = per_q["C3_rag_router"]
        ens.new_task_adapter("task_B")
        ens.new_task_adapter("task_C")
        ems_after = []
        for item in qa:
            out, _, _ = generate_config(
                ens,
                to_ids(item),
                args.n_new,
                use_rag=True,
                use_router=True,
                use_jepa=False,
            )
            ems_after.append(em_score(decode(out), gold_text(item)))
        bt = sum(ems_after) / len(ems_after) - sum(ems_before) / len(
            ems_before
        )
        print(f"  backward transfer (≈0 is ideal): {bt:+.3f}")

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--llm",
        default=None,
        help="HF id / path, 'tiny', or omit for LLM_PATH / ACTIVE_MODEL from .env",
    )
    p.add_argument("--preset", default=None, help="models.yaml preset override")
    p.add_argument("--qa", default=None, help="jsonl with question/answer")
    p.add_argument("--kb", default=None, help="jsonl with text -> vector store")
    p.add_argument(
        "--ckpt",
        default=None,
        help="saved ensemble directory (models/ensemble/... with manifest.json)",
    )
    p.add_argument("--toy", action="store_true", help="synthetic data smoke test")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--n_new", type=int, default=24)
    p.add_argument("--n_drafts", type=int, default=8)
    p.add_argument("--continual", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
