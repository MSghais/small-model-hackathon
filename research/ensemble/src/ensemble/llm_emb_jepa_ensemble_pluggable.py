"""
LLM + Embedding + JEPA Ensemble — pluggable base-model edition
==============================================================
Now the LLM is a swappable BACKEND. Three ways to load it:

    # 1. HuggingFace Hub id
    ens = Ensemble(llm="Qwen/Qwen2.5-0.5B-Instruct")

    # 2. Local path (e.g. downloaded Llama / converted checkpoint)
    ens = Ensemble(llm="/models/llama-3.2-1b")

    # 3. Toy fallback (no transformers needed, runs on CPU in seconds)
    ens = Ensemble(llm="tiny")

Requirements for real models:
    pip install torch transformers peft accelerate
    (optional 4-bit: pip install bitsandbytes -> load_in_4bit=True)

Everything else (Embedder, JEPA, Bridge, VectorStore, Router, the
JEPA-critic inference loop, continual-learning hooks) only touches
token ids / hidden states / latents, so it works with ANY backend.
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# 0. Backend interface — everything the ensemble needs from "an LLM"
# ----------------------------------------------------------------------------
class LLMBackend(nn.Module):
    """Contract:
        vocab_size : int
        hidden_size: int
        device     : torch.device
        forward(ids)            -> (logits [B,T,V], hidden [B,T,H])
        generate(ids, n_new)    -> ids [B, T+n_new]
        add_adapter(name) / set_adapter(name)
        trainable_parameters()  -> iterable of params to optimize
        encode_text(str) / decode(ids)   (real backends only)
    """
    vocab_size: int
    hidden_size: int


# ----------------------------------------------------------------------------
# 0a. HuggingFace backend (local path OR hub id) with PEFT LoRA adapters
# ----------------------------------------------------------------------------
class HFBackend(LLMBackend):
    def __init__(self, model_path: str, *, load_in_4bit: bool = False,
                 lora_r: int = 16, lora_alpha: int = 32,
                 target_modules=("q_proj", "v_proj"),
                 device: str | None = None, torch_dtype=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        self.device_ = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        # `model_path` may be "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/...",
        # or a local directory like "/models/llama-3.2-1b".
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            base.to(self.device_)

        # Freeze the base; all learning happens in LoRA adapters.
        for p in base.parameters():
            p.requires_grad_(False)

        self._lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules=list(target_modules), task_type="CAUSAL_LM")
        self.model = get_peft_model(base, self._lora_cfg, adapter_name="general")
        self._adapters = {"general"}

        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size

    # ---- adapters -----------------------------------------------------------
    def add_adapter(self, name: str):
        if name not in self._adapters:
            self.model.add_adapter(name, self._lora_cfg)
            self._adapters.add(name)

    def set_adapter(self, name: str):
        self.model.set_adapter(name)

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    # ---- core ops -----------------------------------------------------------
    def forward(self, ids):
        out = self.model(input_ids=ids.to(self.device_),
                         output_hidden_states=True)
        return out.logits, out.hidden_states[-1]      # last layer hidden

    @torch.no_grad()
    def generate(self, ids, n_new=64, temperature=0.8):
        out = self.model.generate(
            input_ids=ids.to(self.device_),
            max_new_tokens=n_new, do_sample=True, temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id)
        return out

    # ---- text helpers -------------------------------------------------------
    def encode_text(self, text: str):
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device_)

    def decode(self, ids):
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    @property
    def device(self):
        return self.device_


# ----------------------------------------------------------------------------
# 0b. Tiny fallback backend (no transformers; same toy model as before)
# ----------------------------------------------------------------------------
class TinyBackend(LLMBackend):
    VOCAB, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, LORA_R = 1000, 128, 2, 4, 32, 8

    class _LoRALinear(nn.Module):
        def __init__(self, d_in, d_out, r):
            super().__init__()
            self.base = nn.Linear(d_in, d_out)
            self.base.weight.requires_grad_(False)
            self.base.bias.requires_grad_(False)
            self.adapters, self.active, self.r = nn.ModuleDict(), None, r

        def add_adapter(self, name):
            A = nn.Linear(self.base.in_features, self.r, bias=False)
            B = nn.Linear(self.r, self.base.out_features, bias=False)
            nn.init.zeros_(B.weight)
            self.adapters[name] = nn.Sequential(A, B)

        def forward(self, x):
            y = self.base(x)
            if self.active and self.active in self.adapters:
                y = y + self.adapters[self.active](x)
            return y

    class _Block(nn.Module):
        def __init__(self, D, H, R):
            super().__init__()
            L = TinyBackend._LoRALinear
            self.ln1 = nn.LayerNorm(D)
            self.attn = nn.MultiheadAttention(D, H, batch_first=True)
            self.ln2 = nn.LayerNorm(D)
            self.up, self.down = L(D, 4 * D, R), L(4 * D, D, R)

        def forward(self, x, mask):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            return x + self.down(F.gelu(self.up(self.ln2(x))))

    def __init__(self):
        super().__init__()
        D, V = self.D_MODEL, self.VOCAB
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(self.SEQ_LEN * 4, D)
        self.blocks = nn.ModuleList(
            [self._Block(D, self.N_HEADS, self.LORA_R) for _ in range(self.N_LAYERS)])
        self.ln_f, self.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)
        self.vocab_size, self.hidden_size = V, D
        self.add_adapter("general")
        self.set_adapter("general")

    def add_adapter(self, name):
        for b in self.blocks:
            b.up.add_adapter(name); b.down.add_adapter(name)

    def set_adapter(self, name):
        for b in self.blocks:
            b.up.active = name; b.down.active = name

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        h = self.ln_f(x)
        return self.head(h), h

    @torch.no_grad()
    def generate(self, ids, n_new=16, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(ids[:, -self.SEQ_LEN:])
            nxt = torch.multinomial(F.softmax(logits[:, -1] / temperature, -1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids

    @property
    def device(self):
        return next(self.parameters()).device


def make_backend(llm: str, **kw) -> LLMBackend:
    """'tiny' -> toy model; anything else -> HF hub id or local path."""
    return TinyBackend() if llm == "tiny" else HFBackend(llm, **kw)


# ----------------------------------------------------------------------------
# 1. Embedder — vocab-agnostic (sized from the backend's tokenizer)
#    Swap for a real model: pass embed_fn=lambda txt: sbert.encode(...)
# ----------------------------------------------------------------------------
class Embedder(nn.Module):
    def __init__(self, vocab_size: int, d_emb: int = 64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_emb)
        self.enc = nn.GRU(d_emb, d_emb, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_emb, d_emb)
        self.d_emb = d_emb

    def forward(self, ids):
        h, _ = self.enc(self.tok(ids))
        return F.normalize(self.proj(h.mean(dim=1)), dim=-1)


# ----------------------------------------------------------------------------
# 2. JEPA — vocab-agnostic latent predictor with EMA target encoder
# ----------------------------------------------------------------------------
class _JEPAEncoder(nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.enc = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, d)

    def forward(self, ids):
        h, _ = self.enc(self.tok(ids))
        return self.out(h.mean(dim=1))


class JEPA(nn.Module):
    def __init__(self, vocab_size: int, d_jepa: int = 64, ema_m: float = 0.996):
        super().__init__()
        self.ctx_enc = _JEPAEncoder(vocab_size, d_jepa)
        self.tgt_enc = copy.deepcopy(self.ctx_enc)
        for p in self.tgt_enc.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(d_jepa, 2 * d_jepa), nn.GELU(), nn.Linear(2 * d_jepa, d_jepa))
        self.m, self.d_jepa = ema_m, d_jepa

    @torch.no_grad()
    def ema_update(self):
        for p_t, p_c in zip(self.tgt_enc.parameters(), self.ctx_enc.parameters()):
            p_t.mul_(self.m).add_(p_c.detach(), alpha=1 - self.m)

    def loss(self, seg_ctx, seg_tgt):
        z_hat = self.predictor(self.ctx_enc(seg_ctx))
        with torch.no_grad():
            z_tgt = self.tgt_enc(seg_tgt)
        pred = F.mse_loss(z_hat, z_tgt)
        var_reg = F.relu(1.0 - z_hat.std(dim=0)).mean()   # anti-collapse
        return pred + 0.5 * var_reg

    @torch.no_grad()
    def predict_next_latent(self, seg_ctx):
        return self.predictor(self.ctx_enc(seg_ctx))

    @torch.no_grad()
    def encode(self, seg):
        return self.tgt_enc(seg)


# ----------------------------------------------------------------------------
# 3. Bridge — sized from backend.hidden_size at construction
# ----------------------------------------------------------------------------
class Bridge(nn.Module):
    def __init__(self, d_llm_hidden: int, d_jepa: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_llm_hidden, d_jepa), nn.GELU(), nn.Linear(d_jepa, d_jepa))

    def forward(self, llm_hidden):                       # [B,T,H] -> [B,d_jepa]
        return self.proj(llm_hidden.float().mean(dim=1))

    def info_nce(self, z1, z2, tau=0.07):
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) +
                      F.cross_entropy(logits.t(), labels))


# ----------------------------------------------------------------------------
# 4. Memory + Router
# ----------------------------------------------------------------------------
class VectorStore:
    def __init__(self):
        self.keys, self.values = [], []

    def add(self, emb, payload):
        self.keys.append(emb.squeeze(0).detach().cpu())
        self.values.append(payload)

    def search(self, q, k=2):
        if not self.keys:
            return []
        K = torch.stack(self.keys)
        sims = (q.detach().cpu() @ K.t()).squeeze(0)
        top = sims.topk(min(k, len(self.keys))).indices
        return [self.values[i] for i in top]


class Router(nn.Module):
    def __init__(self, d_emb, n_adapters):
        super().__init__()
        self.fc = nn.Linear(d_emb, n_adapters)

    def forward(self, emb):
        return self.fc(emb).argmax(dim=-1)


# ----------------------------------------------------------------------------
# 5. Ensemble — backend-agnostic
# ----------------------------------------------------------------------------
class Ensemble(nn.Module):
    def __init__(self, llm: str = "tiny", adapter_names=("general",),
                 d_emb: int = 64, d_jepa: int = 64, **backend_kw):
        super().__init__()
        self.llm = make_backend(llm, **backend_kw)
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

    # -------- inference: route -> retrieve -> generate -> JEPA-verify -------
    @torch.no_grad()
    def answer_ids(self, query_ids, n_new=32, tau_consistency=0.0, max_retries=2):
        q_emb = self.emb(query_ids.cpu())
        a_idx = self.router(q_emb).item()
        self.llm.set_adapter(self.adapter_names[a_idx])

        mems = self.store.search(q_emb, k=1)
        ctx = (torch.cat([mems[0], query_ids.cpu()], dim=1)
               if mems else query_ids.cpu())

        z_expected = self.jepa.predict_next_latent(ctx)

        best = None
        for attempt in range(max_retries + 1):
            draft = self.llm.generate(ctx.to(self.llm.device), n_new=n_new,
                                      temperature=0.8 + 0.3 * attempt)
            new_part = draft[:, ctx.size(1):].cpu()
            score = F.cosine_similarity(
                z_expected, self.jepa.encode(new_part)).item()
            if best is None or score > best[1]:
                best = (draft, score, attempt)
            if score >= tau_consistency:
                break
        draft, score, attempt = best
        return draft, score, self.adapter_names[a_idx], attempt

    def answer_text(self, prompt: str, **kw):
        """Convenience wrapper for HF backends (uses the real tokenizer)."""
        ids = self.llm.encode_text(prompt)
        out, score, adapter, retries = self.answer_ids(ids, **kw)
        return self.llm.decode(out), score, adapter, retries

    # -------- continual learning hooks ---------------------------------------
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

    # -------- one joint training step (LM + JEPA + Bridge) -------------------
    def train_step(self, seg_a, seg_b, opt, w_bridge=0.1):
        """seg_a, seg_b: consecutive token-id segments [B, T] (same tokenizer
        as the backend!). For HF backends build them with backend.tokenizer."""
        logits, hidden = self.llm(seg_a.to(self.llm.device))
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.llm.vocab_size).float(),
            seg_a[:, 1:].reshape(-1).to(logits.device))

        jepa_loss = self.jepa.loss(seg_a.cpu(), seg_b.cpu())

        z_llm = self.bridge(hidden.cpu() if hidden.device.type != "cpu" else hidden)
        z_jepa = self.jepa.ctx_enc(seg_a.cpu()).detach()
        bridge_loss = self.bridge.info_nce(z_llm, z_jepa.to(z_llm.device))

        loss = lm_loss.cpu() + jepa_loss + w_bridge * bridge_loss
        opt.zero_grad(); loss.backward(); opt.step()
        self.jepa.ema_update()
        return {"lm": lm_loss.item(), "jepa": jepa_loss.item(),
                "bridge": bridge_loss.item()}

    def make_optimizer(self, lr_lora=2e-4, lr_aux=1e-3):
        return torch.optim.AdamW([
            {"params": list(self.llm.trainable_parameters()), "lr": lr_lora},
            {"params": list(self.jepa.ctx_enc.parameters())
                     + list(self.jepa.predictor.parameters()), "lr": lr_aux},
            {"params": list(self.bridge.parameters())
                     + list(self.emb.parameters())
                     + list(self.router.parameters()), "lr": lr_aux},
        ])


# ----------------------------------------------------------------------------
# 6. Helpers: turn raw text into (seg_a, seg_b) pairs with the HF tokenizer
# ----------------------------------------------------------------------------
def segment_pairs_from_texts(backend: HFBackend, texts, seg_len=64):
    """Yields consecutive-segment id pairs for the JEPA + LM losses."""
    a_list, b_list = [], []
    for t in texts:
        ids = backend.tokenizer(t, return_tensors="pt").input_ids[0]
        for i in range(0, len(ids) - 2 * seg_len, seg_len):
            a_list.append(ids[i:i + seg_len])
            b_list.append(ids[i + seg_len:i + 2 * seg_len])
    if not a_list:
        raise ValueError("texts too short for the chosen seg_len")
    return torch.stack(a_list), torch.stack(b_list)


# ----------------------------------------------------------------------------
# 7. Demos
# ----------------------------------------------------------------------------
def demo_tiny(steps=50):
    """No-dependency smoke test."""
    ens = Ensemble(llm="tiny")
    opt = ens.make_optimizer()
    for s in range(steps):
        seg_a = torch.randint(0, ens.llm.vocab_size, (8, 32))
        seg_b = torch.randint(0, ens.llm.vocab_size, (8, 32))
        logs = ens.train_step(seg_a, seg_b, opt)
        if s % 10 == 0:
            print(f"step {s:3d} | " + " | ".join(f"{k} {v:.3f}" for k, v in logs.items()))

    for _ in range(5):
        ens.memorize_ids(torch.randint(0, ens.llm.vocab_size, (1, 32)))
    ens.new_task_adapter("medical")

    q = torch.randint(0, ens.llm.vocab_size, (1, 8))
    out, score, adapter, retries = ens.answer_ids(q, tau_consistency=-1.0)
    print(f"\nadapter={adapter} jepa_consistency={score:.3f} retries={retries}")


def demo_hf(model_path="Qwen/Qwen2.5-0.5B-Instruct"):
    """Real model from hub id OR local path, e.g. '/models/llama-3.2-1b'.
    For gated Llama repos: huggingface-cli login first."""
    ens = Ensemble(llm=model_path, load_in_4bit=False)   # 4bit needs bitsandbytes
    opt = ens.make_optimizer()

    texts = ["Replace this with your real corpus. " * 50]
    seg_a, seg_b = segment_pairs_from_texts(ens.llm, texts, seg_len=32)
    for s in range(10):                                   # tiny demo run
        logs = ens.train_step(seg_a[:4], seg_b[:4], opt)
        print(f"step {s} | " + " | ".join(f"{k} {v:.3f}" for k, v in logs.items()))

    ens.memorize_text("The project codename is AURORA and it ships in Q3.")
    ens.new_task_adapter("project_aurora")

    text, score, adapter, retries = ens.answer_text(
        "What is the project codename?", n_new=24, tau_consistency=-1.0)
    print(f"\n[{adapter} | jepa={score:.3f} | retries={retries}]\n{text}")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    if arg == "tiny":
        demo_tiny()
    else:
        demo_hf(arg)   # python ensemble.py /models/llama-3.2-1b
                       # python ensemble.py Qwen/Qwen2.5-0.5B-Instruct
