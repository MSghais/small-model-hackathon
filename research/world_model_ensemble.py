"""
World-Model Ensemble: EMB + EBM + JEPA + World Model + small LLM (from path)
=============================================================================
A LeCun-style modular agent built around a small language model.

ARCHITECTURE
------------
                          ┌────────────────────────────┐
   input tokens ──► EMB ──┤ VectorStore (retrieval/CL) │──► context
        │                 └────────────────────────────┘      │
        │                                                     │
        ▼                                                     ▼
   JEPA encoder ──► latent state s_t ──► WORLD MODEL ──► ŝ_{t+1..t+H}
        │                 (GRU dynamics, multi-step rollout)   │
        │                                                      │
        │            ┌────────────────────────────────────┐   │
        └──────────► │ ENERGY MODEL  E(s_ctx, z_candidate)│ ◄─┘
                     │ low energy = compatible/plausible  │
                     └────────────────┬───────────────────┘
                                      │ scores drafts / plans
                                      ▼
   LLM (small, loaded from path, LoRA bank) ──► N drafts ──► pick argmin E

ROLES
-----
EMB         perception for retrieval + routing (non-parametric memory)
JEPA        learns the latent space: predict z(next segment) from z(context)
            (EMA target encoder + variance reg, no token reconstruction)
WORLD MODEL deterministic latent dynamics  s_{t+1} = f(s_t, z_t):
            rolls the conversation/document state forward H steps in
            LATENT space — cheap lookahead without decoding tokens
ENERGY      E(s, z) ∈ R, trained so true continuations have LOW energy and
            negatives (shuffled / model-generated) have HIGH energy.
            At inference it is the critic: rank LLM drafts, reject bad plans.
LLM         the only token-level generator. Loaded from a local path or HF id;
            frozen base + LoRA adapters (continual learning by isolation).

WHY EBM *and* JEPA?  JEPA gives a point prediction ẑ of the future latent;
the EBM gives a *compatibility landscape* E(s, z) — it can say "both A and B
are plausible" where a point predictor must average them. JEPA trains the
representation; the EBM scores hypotheses in it. World model chains JEPA
one-step predictions into multi-step rollouts that the EBM can evaluate.

USAGE
-----
    pip install torch            # toy mode
    pip install transformers peft accelerate   # real LLM mode

    python world_model_ensemble.py tiny                 # smoke test
    python world_model_ensemble.py /models/llama-3.2-1b # local weights
    python world_model_ensemble.py Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations
import copy
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D_LAT = 96          # shared latent dimension (JEPA / world / energy)
D_EMB = 64          # retrieval embedding dim


# ============================================================================
# 1. LLM backend — load small model from path / hub, or toy fallback
#    (same contract as before: forward -> (logits, hidden), generate, adapters)
# ============================================================================
class TinyLLM(nn.Module):
    VOCAB, D, L, H, T = 1000, 128, 2, 4, 32

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(self.VOCAB, self.D)
        self.pos = nn.Embedding(self.T * 4, self.D)
        layer = nn.TransformerEncoderLayer(self.D, self.H, 4 * self.D,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, self.L)
        self.head = nn.Linear(self.D, self.VOCAB, bias=False)
        self.vocab_size, self.hidden_size = self.VOCAB, self.D

    def forward(self, ids):
        Tn = ids.size(1)
        x = self.tok(ids) + self.pos(torch.arange(Tn, device=ids.device))
        mask = torch.triu(torch.full((Tn, Tn), float("-inf"),
                                     device=ids.device), 1)
        h = self.blocks(x, mask=mask)
        return self.head(h), h

    @torch.no_grad()
    def generate(self, ids, n_new=16, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(ids[:, -self.T:])
            nxt = torch.multinomial(
                F.softmax(logits[:, -1] / temperature, -1), 1)
            ids = torch.cat([ids, nxt], 1)
        return ids

    def trainable_parameters(self):
        return self.parameters()

    @property
    def device(self):
        return next(self.parameters()).device


class HFLLM(nn.Module):
    """Small model from a local path or HF id, frozen base + LoRA."""
    def __init__(self, path, lora_r=16):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16
            if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None)
        for p in base.parameters():
            p.requires_grad_(False)
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                         target_modules=["q_proj", "v_proj"],
                         task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size

    def forward(self, ids):
        out = self.model(input_ids=ids.to(self.device),
                         output_hidden_states=True)
        return out.logits, out.hidden_states[-1]

    @torch.no_grad()
    def generate(self, ids, n_new=32, temperature=0.8):
        return self.model.generate(
            input_ids=ids.to(self.device), max_new_tokens=n_new,
            do_sample=True, temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id)

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.model.parameters()).device


def load_llm(spec: str):
    return TinyLLM() if spec == "tiny" else HFLLM(spec)


# ============================================================================
# 2. Embedder (retrieval) — vocab-agnostic
# ============================================================================
class Embedder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_EMB)
        self.gru = nn.GRU(D_EMB, D_EMB, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * D_EMB, D_EMB)

    def forward(self, ids):
        h, _ = self.gru(self.tok(ids))
        return F.normalize(self.out(h.mean(1)), dim=-1)


class VectorStore:
    def __init__(self):
        self.K, self.V = [], []

    def add(self, k, v):
        self.K.append(k.squeeze(0).detach().cpu()); self.V.append(v)

    def search(self, q, k=1):
        if not self.K:
            return []
        sims = (q.detach().cpu() @ torch.stack(self.K).t()).squeeze(0)
        return [self.V[i] for i in sims.topk(min(k, len(self.K))).indices]


# ============================================================================
# 3. JEPA — owns the latent space (EMA target encoder, variance-regularized)
# ============================================================================
class SegEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_LAT)
        self.gru = nn.GRU(D_LAT, D_LAT, batch_first=True)
        self.out = nn.Linear(D_LAT, D_LAT)

    def forward(self, ids):
        h, _ = self.gru(self.tok(ids))
        return self.out(h.mean(1))                       # [B, D_LAT]


class JEPA(nn.Module):
    def __init__(self, vocab, m=0.996):
        super().__init__()
        self.enc = SegEncoder(vocab)                     # context/online enc
        self.tgt = copy.deepcopy(self.enc)               # EMA target
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.pred = nn.Sequential(nn.Linear(D_LAT, 2 * D_LAT), nn.GELU(),
                                  nn.Linear(2 * D_LAT, D_LAT))
        self.m = m

    @torch.no_grad()
    def ema(self):
        for pt, pc in zip(self.tgt.parameters(), self.enc.parameters()):
            pt.mul_(self.m).add_(pc.detach(), alpha=1 - self.m)

    def loss(self, seg_a, seg_b):
        z_hat = self.pred(self.enc(seg_a))
        with torch.no_grad():
            z_tgt = self.tgt(seg_b)
        var = F.relu(1.0 - z_hat.std(0)).mean()          # anti-collapse
        return F.mse_loss(z_hat, z_tgt) + 0.5 * var

    @torch.no_grad()
    def encode(self, seg):                               # target space
        return self.tgt(seg)


# ============================================================================
# 4. WORLD MODEL — latent dynamics s_{t+1} = f(s_t, z_t), multi-step rollout
#    Trained on SEQUENCES of segments: predict each next latent from state.
# ============================================================================
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(D_LAT, D_LAT)             # state update
        self.head = nn.Linear(D_LAT, D_LAT)              # state -> ẑ_{t+1}
        self.s0 = nn.Parameter(torch.zeros(D_LAT))

    def init_state(self, B, device):
        return self.s0.unsqueeze(0).expand(B, -1).contiguous().to(device)

    def step(self, s, z):
        """Consume observed latent z_t, return (new state, prediction ẑ_{t+1})."""
        s = self.cell(z, s)
        return s, self.head(s)

    def rollout(self, s, horizon):
        """Imagine H future latents feeding its own predictions back in."""
        preds = []
        for _ in range(horizon):
            z_hat = self.head(s)
            preds.append(z_hat)
            s = self.cell(z_hat, s)
        return torch.stack(preds, 1), s                  # [B, H, D_LAT]

    def sequence_loss(self, z_seq):
        """z_seq: [B, T, D_LAT] observed segment latents (teacher forcing)."""
        B, T, _ = z_seq.shape
        s = self.init_state(B, z_seq.device)
        loss = 0.0
        for t in range(T - 1):
            s, z_hat = self.step(s, z_seq[:, t])
            loss = loss + F.mse_loss(z_hat, z_seq[:, t + 1])
        return loss / (T - 1)


# ============================================================================
# 5. ENERGY MODEL — E(state, candidate latent) ∈ R, low = plausible
#    Trained with InfoNCE-style contrastive: positives = true next latent,
#    negatives = (a) other batch items, (b) LLM-generated drafts (optional).
# ============================================================================
class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * D_LAT, 2 * D_LAT), nn.GELU(),
            nn.Linear(2 * D_LAT, D_LAT), nn.GELU(),
            nn.Linear(D_LAT, 1))

    def energy(self, s, z):
        """s: [B, D_LAT] context state; z: [B, D_LAT] candidate. -> [B]"""
        return self.net(torch.cat([s, z], -1)).squeeze(-1)

    def contrastive_loss(self, s, z_pos, z_negs=None, tau=0.5):
        """Softmax over energies: true continuation must be the argmin.
        In-batch negatives: every other item's z_pos is a negative for s_i."""
        B = s.size(0)
        # pairwise energies: E(s_i, z_j) for all i, j
        s_rep = s.unsqueeze(1).expand(B, B, D_LAT).reshape(B * B, D_LAT)
        z_rep = z_pos.unsqueeze(0).expand(B, B, D_LAT).reshape(B * B, D_LAT)
        E = self.energy(s_rep, z_rep).view(B, B)         # [B, B]
        if z_negs is not None:                           # extra hard negatives
            En = self.energy(
                s.repeat_interleave(z_negs.size(1), 0),
                z_negs.reshape(-1, D_LAT)).view(B, -1)
            E = torch.cat([E, En], dim=1)
        labels = torch.arange(B, device=s.device)
        return F.cross_entropy(-E / tau, labels)         # low E ⇒ high logit

    @torch.no_grad()
    def rank(self, s, candidates):
        """candidates: [N, D_LAT]; returns energies [N] (lower = better)."""
        return self.energy(s.expand(candidates.size(0), -1), candidates)


# ============================================================================
# 6. Bridge — LLM hidden states -> shared latent space (alignment)
# ============================================================================
class Bridge(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_hidden, D_LAT), nn.GELU(),
                                  nn.Linear(D_LAT, D_LAT))

    def forward(self, h):                                # [B,T,H] -> [B,D_LAT]
        return self.proj(h.float().mean(1))

    def info_nce(self, a, b, tau=0.07):
        a, b = F.normalize(a, -1), F.normalize(b, -1)
        logits = a @ b.t() / tau
        y = torch.arange(a.size(0), device=a.device)
        return 0.5 * (F.cross_entropy(logits, y) +
                      F.cross_entropy(logits.t(), y))


# ============================================================================
# 7. THE ENSEMBLE — wiring + inference (plan -> generate -> energy-rank)
# ============================================================================
class WorldEnsemble(nn.Module):
    def __init__(self, llm_spec="tiny"):
        super().__init__()
        self.llm = load_llm(llm_spec)
        V, H = self.llm.vocab_size, self.llm.hidden_size
        self.emb = Embedder(V)
        self.jepa = JEPA(V)
        self.world = WorldModel()
        self.energy = EnergyModel()
        self.bridge = Bridge(H)
        self.store = VectorStore()

    # ------------------------- inference ---------------------------------
    @torch.no_grad()
    def world_state(self, segments):
        """Fold a list of [1,T] segment tensors into a latent state."""
        s = self.world.init_state(1, "cpu")
        for seg in segments:
            z = self.jepa.encode(seg.cpu())
            s, _ = self.world.step(s, z)
        return s

    @torch.no_grad()
    def answer(self, query_ids, n_new=24, n_drafts=6, horizon=3):
        """retrieve -> build world state -> imagine -> generate N -> argmin E."""
        q_emb = self.emb(query_ids.cpu())
        mems = self.store.search(q_emb, k=1)
        segments = (mems + [query_ids.cpu()]) if mems else [query_ids.cpu()]
        ctx = torch.cat(segments, dim=1)

        s = self.world_state(segments)                   # latent context state
        plan, _ = self.world.rollout(s, horizon)         # imagined future
        # (plan is available for planning losses / steering; logged here)

        drafts, lat = [], []
        for _ in range(n_drafts):
            out = self.llm.generate(ctx.to(self.llm.device), n_new=n_new,
                                    temperature=0.9)
            new = out[:, ctx.size(1):].cpu()
            drafts.append(new)
            lat.append(self.jepa.encode(new))
        Z = torch.cat(lat, 0)                            # [N, D_LAT]
        E = self.energy.rank(s, Z)                       # lower = better
        best = E.argmin().item()
        return {"output": drafts[best], "energy": E[best].item(),
                "all_energies": E.tolist(),
                "plan_alignment": F.cosine_similarity(
                    plan[:, 0], Z[best:best + 1]).item()}

    def memorize(self, ids):
        self.store.add(self.emb(ids.cpu()), ids.cpu())

    # ------------------------- training ----------------------------------
    def train_step(self, seg_seq, opt, w=dict(lm=1.0, jepa=1.0, world=1.0,
                                              ebm=1.0, bridge=0.1),
                   hard_negs=True):
        """seg_seq: [B, T_seg, L] — B documents, each split into T_seg
        consecutive segments of length L (same tokenizer as the LLM)."""
        B, T, L = seg_seq.shape
        dev = self.llm.device

        # (1) LM loss on the first segment (or all, batched, if budget allows)
        flat = seg_seq[:, 0].to(dev)
        logits, hidden = self.llm(flat)
        lm = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.llm.vocab_size).float(),
            flat[:, 1:].reshape(-1))

        # (2) JEPA: adjacent segment pairs
        jepa = self.jepa.loss(seg_seq[:, 0], seg_seq[:, 1])

        # (3) World model: sequence of latents (online encoder, grads flow)
        z_seq = torch.stack([self.jepa.enc(seg_seq[:, t])
                             for t in range(T)], 1)      # [B, T, D_LAT]
        world = self.world.sequence_loss(z_seq)

        # (4) Energy: state after t=0 must give low E to true z_1,
        #     high E to in-batch + (optionally) LLM-generated negatives
        s = self.world.init_state(B, z_seq.device)
        s, _ = self.world.step(s, z_seq[:, 0].detach())
        z_pos = z_seq[:, 1].detach()
        z_negs = None
        if hard_negs:
            with torch.no_grad():                        # model drafts as negs
                gen = self.llm.generate(seg_seq[:, 0].to(dev), n_new=L)
                gen_new = gen[:, seg_seq.size(2):].cpu()
                z_negs = self.jepa.encode(gen_new).unsqueeze(1)  # [B,1,D]
        ebm = self.energy.contrastive_loss(s, z_pos, z_negs)

        # (5) Bridge: align LLM hidden(seg0) with JEPA latent(seg0)
        bridge = self.bridge.info_nce(
            self.bridge(hidden.cpu() if hidden.device.type != "cpu" else hidden),
            self.jepa.enc(seg_seq[:, 0]).detach())

        loss = (w["lm"] * lm.cpu() + w["jepa"] * jepa + w["world"] * world
                + w["ebm"] * ebm + w["bridge"] * bridge)
        opt.zero_grad(); loss.backward(); opt.step()
        self.jepa.ema()
        return dict(lm=lm.item(), jepa=jepa.item(), world=world.item(),
                    ebm=ebm.item(), bridge=bridge.item())

    def make_optimizer(self, lr_lora=2e-4, lr_aux=1e-3):
        return torch.optim.AdamW([
            {"params": list(self.llm.trainable_parameters()), "lr": lr_lora},
            {"params": list(self.jepa.enc.parameters())
                     + list(self.jepa.pred.parameters()), "lr": lr_aux},
            {"params": list(self.world.parameters()), "lr": lr_aux},
            {"params": list(self.energy.parameters()), "lr": lr_aux},
            {"params": list(self.bridge.parameters())
                     + list(self.emb.parameters()), "lr": lr_aux}])


# ============================================================================
# 8. Data helpers + demo
# ============================================================================
def toy_segment_sequences(B=8, T=4, L=24, vocab=1000):
    """Random docs split into T consecutive segments. Replace with real
    corpus: tokenize each document, reshape into [T, L] windows."""
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
        logs = ens.train_step(get_batch(), opt,
                              hard_negs=(s > steps // 2))  # warmup w/o negs
        if s % 10 == 0:
            print(f"step {s:3d} | " +
                  " | ".join(f"{k} {v:.3f}" for k, v in logs.items()))
    print(f"trained {steps} steps in {time.time()-t0:.1f}s")

    # memory + inference
    for _ in range(4):
        if spec == "tiny":
            ens.memorize(torch.randint(0, ens.llm.vocab_size, (1, 24)))
    q = (torch.randint(0, ens.llm.vocab_size, (1, 12)) if spec == "tiny"
         else ens.llm.tokenizer("What is this document about?",
                                return_tensors="pt").input_ids)
    res = ens.answer(q, n_drafts=6, horizon=3)
    print(f"\nselected draft energy={res['energy']:.3f} "
          f"(all: {[f'{e:.2f}' for e in res['all_energies']]})")
    print(f"plan↔output alignment: {res['plan_alignment']:.3f}")


if __name__ == "__main__":
    demo(sys.argv[1] if len(sys.argv) > 1 else "tiny")
