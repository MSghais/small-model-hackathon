"""JEPA ensemble: route -> retrieve -> generate -> JEPA-verify."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ensemble.backends import HFBackend, make_backend
from ensemble.bridge import Bridge
from ensemble.jepa import JEPA
from ensemble.memory import Embedder, Router, VectorStore

torch.manual_seed(0)


class Ensemble(nn.Module):
    def __init__(
        self,
        llm: str = "tiny",
        adapter_names=("general",),
        d_emb: int = 64,
        d_jepa: int = 64,
        llm_backend: HFBackend | None = None,
        **backend_kw,
    ):
        super().__init__()
        self.llm = llm_backend if llm_backend is not None else make_backend(llm, **backend_kw)
        V, H = self.llm.vocab_size, self.llm.hidden_size

        self.emb = Embedder(V, d_emb)
        self.jepa = JEPA(V, d_jepa)
        self.bridge = Bridge(H, d_jepa)
        self.store = VectorStore()

        self.adapter_names = list(adapter_names)
        for n in self.adapter_names:
            self.llm.add_adapter(n)
        self.llm.set_adapter(self.adapter_names[0])
        self.router = Router(d_emb, len(self.adapter_names))

    @torch.no_grad()
    def answer_ids(
        self,
        query_ids,
        n_new=32,
        tau_consistency=0.0,
        max_retries=2,
        temperature: float = 0.7,
    ):
        q_emb = self.emb(query_ids.cpu())
        a_idx = self.router(q_emb).item()
        self.llm.set_adapter(self.adapter_names[a_idx])

        mems = self.store.search(q_emb, k=1)
        ctx = (
            torch.cat([mems[0], query_ids.cpu()], dim=1)
            if mems
            else query_ids.cpu()
        )

        z_expected = self.jepa.predict_next_latent(ctx)

        best = None
        for attempt in range(max_retries + 1):
            temp = temperature if attempt == 0 else max(temperature, 0.8 + 0.3 * attempt)
            draft = self.llm.generate(
                ctx.to(self.llm.device),
                n_new=n_new,
                temperature=temp,
            )
            new_part = draft[:, ctx.size(1) :].cpu()
            score = F.cosine_similarity(
                z_expected, self.jepa.encode(new_part)
            ).item()
            if best is None or score > best[1]:
                best = (draft, score, attempt)
            if score >= tau_consistency:
                break
        draft, score, attempt = best
        return draft, score, self.adapter_names[a_idx], attempt

    def answer_text(self, prompt: str, **kw):
        ids = self.llm.encode_text(prompt)
        out, score, adapter, retries = self.answer_ids(ids, **kw)
        return self.llm.decode(out), score, adapter, retries

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Greedy or sampled generation through the full ensemble stack."""
        ids = self.llm.encode_text(prompt)
        out, _, _, _ = self.answer_ids(
            ids,
            n_new=max_new_tokens,
            tau_consistency=-1.0,
            max_retries=0 if temperature <= 0 else 1,
            temperature=temperature,
        )
        return self.llm.decode(out)

    def memorize_ids(self, ids):
        self.store.add(self.emb(ids.cpu()), ids.cpu())

    def memorize_text(self, text: str):
        self.memorize_ids(self.llm.encode_text(text))

    def new_task_adapter(self, name: str):
        self.adapter_names.append(name)
        self.llm.add_adapter(name)
        old = self.router
        self.router = Router(self.emb.d_emb, len(self.adapter_names))
        with torch.no_grad():
            self.router.fc.weight[: old.fc.out_features] = old.fc.weight
            self.router.fc.bias[: old.fc.out_features] = old.fc.bias

    def train_step(self, seg_a, seg_b, opt, w_bridge=0.1):
        logits, hidden = self.llm(seg_a.to(self.llm.device))
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.llm.vocab_size).float(),
            seg_a[:, 1:].reshape(-1).to(logits.device),
        )

        jepa_loss = self.jepa.loss(seg_a.cpu(), seg_b.cpu())

        z_llm = self.bridge(
            hidden.cpu() if hidden.device.type != "cpu" else hidden
        )
        z_jepa = self.jepa.ctx_enc(seg_a.cpu()).detach()
        bridge_loss = self.bridge.info_nce(z_llm, z_jepa.to(z_llm.device))

        loss = lm_loss.cpu() + jepa_loss + w_bridge * bridge_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        self.jepa.ema_update()
        return {
            "lm": lm_loss.item(),
            "jepa": jepa_loss.item(),
            "bridge": bridge_loss.item(),
        }

    def make_optimizer(self, lr_lora=2e-4, lr_aux=1e-3):
        return torch.optim.AdamW(
            [
                {"params": list(self.llm.trainable_parameters()), "lr": lr_lora},
                {
                    "params": list(self.jepa.ctx_enc.parameters())
                    + list(self.jepa.predictor.parameters()),
                    "lr": lr_aux,
                },
                {
                    "params": list(self.bridge.parameters())
                    + list(self.emb.parameters())
                    + list(self.router.parameters()),
                    "lr": lr_aux,
                },
            ]
        )


def segment_pairs_from_texts(backend: HFBackend, texts, seg_len=64):
    a_list, b_list = [], []
    for t in texts:
        ids = backend.tokenizer(t, return_tensors="pt").input_ids[0]
        for i in range(0, len(ids) - 2 * seg_len, seg_len):
            a_list.append(ids[i : i + seg_len])
            b_list.append(ids[i + seg_len : i + 2 * seg_len])
    if not a_list:
        raise ValueError("texts too short for the chosen seg_len")
    return torch.stack(a_list), torch.stack(b_list)


def demo_tiny(steps=50):
    ens = Ensemble(llm="tiny")
    opt = ens.make_optimizer()
    for s in range(steps):
        seg_a = torch.randint(0, ens.llm.vocab_size, (8, 32))
        seg_b = torch.randint(0, ens.llm.vocab_size, (8, 32))
        logs = ens.train_step(seg_a, seg_b, opt)
        if s % 10 == 0:
            print(
                f"step {s:3d} | "
                + " | ".join(f"{k} {v:.3f}" for k, v in logs.items())
            )

    for _ in range(5):
        ens.memorize_ids(torch.randint(0, ens.llm.vocab_size, (1, 32)))
    ens.new_task_adapter("medical")

    q = torch.randint(0, ens.llm.vocab_size, (1, 8))
    out, score, adapter, retries = ens.answer_ids(q, tau_consistency=-1.0)
    print(f"\nadapter={adapter} jepa_consistency={score:.3f} retries={retries}")


def demo_hf(model_path="Qwen/Qwen2.5-0.5B-Instruct"):
    ens = Ensemble(llm=model_path, load_in_4bit=False)
    opt = ens.make_optimizer()

    texts = ["Replace this with your real corpus. " * 50]
    seg_a, seg_b = segment_pairs_from_texts(ens.llm, texts, seg_len=32)
    for s in range(10):
        logs = ens.train_step(seg_a[:4], seg_b[:4], opt)
        print(f"step {s} | " + " | ".join(f"{k} {v:.3f}" for k, v in logs.items()))

    ens.memorize_text("The project codename is AURORA and it ships in Q3.")
    ens.new_task_adapter("project_aurora")

    text, score, adapter, retries = ens.answer_text(
        "What is the project codename?", n_new=24, tau_consistency=-1.0
    )
    print(f"\n[{adapter} | jepa={score:.3f} | retries={retries}]\n{text}")


if __name__ == "__main__":
    import sys

    from ensemble.config import load_dotenv, resolve_llm

    load_dotenv()
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None or arg == "auto":
        arg, preset = resolve_llm()
        print(f"Resolved LLM: {arg} (preset {preset})")
    if arg == "tiny":
        demo_tiny()
    else:
        demo_hf(arg)
