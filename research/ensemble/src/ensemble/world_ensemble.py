"""World-model ensemble: plan -> generate -> energy-rank."""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ensemble.backends import HFLLM, load_llm
from ensemble.bridge import Bridge
from ensemble.energy import EnergyModel
from ensemble.jepa import JEPA
from ensemble.memory import Embedder, VectorStore
from ensemble.world_model import WorldModel

torch.manual_seed(0)

D_LAT = 96
D_EMB = 64


class WorldEnsemble(nn.Module):
    def __init__(self, llm_spec="tiny"):
        super().__init__()
        self.llm = load_llm(llm_spec)
        V, H = self.llm.vocab_size, self.llm.hidden_size
        self.emb = Embedder(V, D_EMB)
        self.jepa = JEPA(V, D_LAT)
        self.world = WorldModel(D_LAT)
        self.energy = EnergyModel(D_LAT)
        self.bridge = Bridge(H, D_LAT)
        self.store = VectorStore()

    @torch.no_grad()
    def world_state(self, segments):
        s = self.world.init_state(1, "cpu")
        for seg in segments:
            z = self.jepa.encode(seg.cpu())
            s, _ = self.world.step(s, z)
        return s

    @torch.no_grad()
    def answer(self, query_ids, n_new=24, n_drafts=6, horizon=3):
        q_emb = self.emb(query_ids.cpu())
        mems = self.store.search(q_emb, k=1)
        segments = (mems + [query_ids.cpu()]) if mems else [query_ids.cpu()]
        ctx = torch.cat(segments, dim=1)

        s = self.world_state(segments)
        plan, _ = self.world.rollout(s, horizon)

        drafts, lat = [], []
        for _ in range(n_drafts):
            out = self.llm.generate(
                ctx.to(self.llm.device), n_new=n_new, temperature=0.9
            )
            new = out[:, ctx.size(1) :].cpu()
            drafts.append(new)
            lat.append(self.jepa.encode(new))
        Z = torch.cat(lat, 0)
        E = self.energy.rank(s, Z)
        best = E.argmin().item()
        return {
            "output": drafts[best],
            "energy": E[best].item(),
            "all_energies": E.tolist(),
            "plan_alignment": F.cosine_similarity(
                plan[:, 0], Z[best : best + 1]
            ).item(),
        }

    def memorize(self, ids):
        self.store.add(self.emb(ids.cpu()), ids.cpu())

    def train_step(
        self,
        seg_seq,
        opt,
        w=None,
        hard_negs=True,
    ):
        if w is None:
            w = dict(lm=1.0, jepa=1.0, world=1.0, ebm=1.0, bridge=0.1)

        B, T, L = seg_seq.shape
        dev = self.llm.device

        flat = seg_seq[:, 0].to(dev)
        logits, hidden = self.llm(flat)
        lm = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.llm.vocab_size).float(),
            flat[:, 1:].reshape(-1),
        )

        jepa = self.jepa.loss(seg_seq[:, 0], seg_seq[:, 1])

        z_seq = torch.stack(
            [self.jepa.enc(seg_seq[:, t]) for t in range(T)], 1
        )
        world = self.world.sequence_loss(z_seq)

        s = self.world.init_state(B, z_seq.device)
        s, _ = self.world.step(s, z_seq[:, 0].detach())
        z_pos = z_seq[:, 1].detach()
        z_negs = None
        if hard_negs:
            with torch.no_grad():
                gen = self.llm.generate(seg_seq[:, 0].to(dev), n_new=L)
                gen_new = gen[:, seg_seq.size(2) :].cpu()
                z_negs = self.jepa.encode(gen_new).unsqueeze(1)
        ebm = self.energy.contrastive_loss(s, z_pos, z_negs)

        bridge = self.bridge.info_nce(
            self.bridge(
                hidden.cpu() if hidden.device.type != "cpu" else hidden
            ),
            self.jepa.enc(seg_seq[:, 0]).detach(),
        )

        loss = (
            w["lm"] * lm.cpu()
            + w["jepa"] * jepa
            + w["world"] * world
            + w["ebm"] * ebm
            + w["bridge"] * bridge
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        self.jepa.ema()
        return dict(
            lm=lm.item(),
            jepa=jepa.item(),
            world=world.item(),
            ebm=ebm.item(),
            bridge=bridge.item(),
        )

    def make_optimizer(self, lr_lora=2e-4, lr_aux=1e-3):
        return torch.optim.AdamW(
            [
                {"params": list(self.llm.trainable_parameters()), "lr": lr_lora},
                {
                    "params": list(self.jepa.enc.parameters())
                    + list(self.jepa.pred.parameters()),
                    "lr": lr_aux,
                },
                {"params": list(self.world.parameters()), "lr": lr_aux},
                {"params": list(self.energy.parameters()), "lr": lr_aux},
                {
                    "params": list(self.bridge.parameters())
                    + list(self.emb.parameters()),
                    "lr": lr_aux,
                },
            ]
        )


def toy_segment_sequences(B=8, T=4, L=24, vocab=1000):
    return torch.randint(0, vocab, (B, T, L))


def hf_segment_sequences(llm: HFLLM, texts, T=4, L=64):
    seqs = []
    for t in texts:
        ids = llm.tokenizer(t, return_tensors="pt").input_ids[0]
        n = (len(ids) // (T * L)) * T * L
        if n:
            seqs.append(ids[:n].view(-1, T, L))
    if not seqs:
        raise ValueError("corpus too short for T*L window")
    return torch.cat(seqs, 0)


def demo(spec="tiny", steps=60):
    ens = WorldEnsemble(spec)
    opt = ens.make_optimizer()

    if spec == "tiny":
        get_batch = lambda: toy_segment_sequences(vocab=ens.llm.vocab_size)
    else:
        corpus = ["Replace with your real documents. " * 200]
        data = hf_segment_sequences(ens.llm, corpus, T=4, L=32)
        get_batch = lambda: data[torch.randperm(len(data))[:4]]
        steps = min(steps, 10)

    t0 = time.time()
    for s in range(steps):
        logs = ens.train_step(
            get_batch(), opt, hard_negs=(s > steps // 2)
        )
        if s % 10 == 0:
            print(
                f"step {s:3d} | "
                + " | ".join(f"{k} {v:.3f}" for k, v in logs.items())
            )
    print(f"trained {steps} steps in {time.time() - t0:.1f}s")

    for _ in range(4):
        if spec == "tiny":
            ens.memorize(torch.randint(0, ens.llm.vocab_size, (1, 24)))
    q = (
        torch.randint(0, ens.llm.vocab_size, (1, 12))
        if spec == "tiny"
        else ens.llm.tokenizer(
            "What is this document about?", return_tensors="pt"
        ).input_ids
    )
    res = ens.answer(q, n_drafts=6, horizon=3)
    print(
        f"\nselected draft energy={res['energy']:.3f} "
        f"(all: {[f'{e:.2f}' for e in res['all_energies']]})"
    )
    print(f"plan↔output alignment: {res['plan_alignment']:.3f}")


if __name__ == "__main__":
    from ensemble.config import load_dotenv, resolve_llm

    load_dotenv()
    spec = sys.argv[1] if len(sys.argv) > 1 else None
    if spec is None or spec == "auto":
        spec, preset = resolve_llm()
        print(f"Resolved LLM: {spec} (preset {preset})")
    demo(spec or "tiny")
