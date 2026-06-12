"""Energy-based draft selector benchmark for the world-model ensemble."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import torch

from ensemble.eval.metrics import em_score, f1_score, paired_bootstrap
from ensemble.world_ensemble import WorldEnsemble


@torch.no_grad()
def generate_drafts(ens, q_ids, n_new, n_drafts, use_rag=True):
    q_emb = ens.emb(q_ids.cpu())
    mems = ens.store.search(q_emb, k=1) if use_rag else []
    segments = (mems + [q_ids.cpu()]) if mems else [q_ids.cpu()]
    ctx = torch.cat(segments, dim=1)

    s = ens.world_state(segments)
    ens.world.rollout(s, horizon=3)

    drafts, energies = [], []
    t0 = time.time()
    for _ in range(n_drafts):
        out = ens.llm.generate(
            ctx.to(ens.llm.device), n_new=n_new, temperature=0.9
        )
        new = out[:, ctx.size(1) :].cpu()
        drafts.append(new)
        z = ens.jepa.encode(new)
        energies.append(ens.energy.rank(s, z).item())
    return drafts, energies, time.time() - t0


def selector_comparison(drafts_energy_gold, decode_fn, rng):
    res = defaultdict(list)
    for drafts, energies, gold in drafts_energy_gold:
        texts = [decode_fn(d) for d in drafts]
        ems = [em_score(t, gold) for t in texts]
        res["first"].append(ems[0])
        res["random"].append(ems[rng.randrange(len(ems))])
        res["energy"].append(
            ems[min(range(len(ems)), key=lambda i: energies[i])]
        )
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
    from ensemble.config import load_dotenv, resolve_llm_cli

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    load_dotenv()
    args.llm = resolve_llm_cli(
        args.llm, toy=args.toy, preset=getattr(args, "preset", None)
    )
    print(f"Resolved LLM: {args.llm}")
    ens = WorldEnsemble(args.llm)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        ens.load_state_dict(state, strict=False)
        print(f"loaded world ensemble checkpoint: {args.ckpt}")

    is_text = args.llm != "tiny"

    if args.toy or not is_text:
        qa, kb = make_toy_data(ens)
        for mem in kb:
            ens.memorize(mem)

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
                ids = ens.llm.tokenizer(
                    row["text"], return_tensors="pt"
                ).input_ids
                ens.memorize(ids)

        def to_ids(item):
            return ens.llm.tokenizer(
                f"Answer briefly.\nQ: {item['question']}\nA:",
                return_tensors="pt",
            ).input_ids

        def gold_text(item):
            return item["answer"]

        def decode(ids):
            return ens.llm.tokenizer.decode(ids[0], skip_special_tokens=True)

    qa = qa[: args.limit]
    print(
        f"eval set: {len(qa)} questions | store: {len(ens.store.keys)} memories\n"
    )

    material = []
    lats = []
    for item in qa:
        drafts, energies, dt = generate_drafts(
            ens, to_ids(item), args.n_new, args.n_drafts
        )
        material.append((drafts, energies, gold_text(item)))
        lats.append(dt)

    sel, sel_per_q = selector_comparison(material, decode, rng)
    print(f"best-of-N selector comparison (same drafts, N={args.n_drafts}):")
    for k in ("first", "random", "energy", "oracle"):
        print(f"  {k:<8}EM={sel[k]:.3f}")
    p = paired_bootstrap(sel_per_q["random"], sel_per_q["energy"])
    verdict = (
        "Energy critic WORKS"
        if p > 0.95
        else "inconclusive — critic ~ random"
    )
    print(f"  P(energy > random) = {p:.2f}   {verdict}")
    print(f"  headroom to oracle: {sel['oracle'] - sel['energy']:.3f}")
    print(f"  mean latency: {sum(lats) / len(lats):.3f}s")

    return sel


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
    p.add_argument("--ckpt", default=None, help="trained world ensemble .pt")
    p.add_argument("--toy", action="store_true", help="synthetic data smoke test")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--n_new", type=int, default=24)
    p.add_argument("--n_drafts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
