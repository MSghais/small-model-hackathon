"""LLM backends: toy fallbacks and HuggingFace + LoRA loaders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMBackend(nn.Module):
    """Contract for JEPA ensemble backends."""

    vocab_size: int
    hidden_size: int


class HFBackend(LLMBackend):
    """HuggingFace causal LM with PEFT LoRA adapter bank."""

    def __init__(
        self,
        model_path: str,
        *,
        load_in_4bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        target_modules=("q_proj", "v_proj"),
        device: str | None = None,
        torch_dtype=None,
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device_ = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            base.to(self.device_)

        for p in base.parameters():
            p.requires_grad_(False)

        self._lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=list(target_modules),
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, self._lora_cfg, adapter_name="general")
        self._adapters = {"general"}

        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size

    def add_adapter(self, name: str):
        if name not in self._adapters:
            self.model.add_adapter(name, self._lora_cfg)
            self._adapters.add(name)

    def set_adapter(self, name: str):
        self.model.set_adapter(name)

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    def forward(self, ids):
        out = self.model(
            input_ids=ids.to(self.device_), output_hidden_states=True
        )
        return out.logits, out.hidden_states[-1]

    @torch.no_grad()
    def generate(self, ids, n_new=64, temperature=0.8):
        gen_kwargs: dict = dict(
            input_ids=ids.to(self.device_),
            max_new_tokens=n_new,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        out = self.model.generate(**gen_kwargs)
        return out

    def encode_text(self, text: str):
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device_)

    def decode(self, ids):
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    @property
    def device(self):
        return self.device_


class TinyBackend(LLMBackend):
    """Toy transformer with LoRA adapters (no transformers dependency)."""

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
            [self._Block(D, self.N_HEADS, self.LORA_R) for _ in range(self.N_LAYERS)]
        )
        self.ln_f, self.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)
        self.vocab_size, self.hidden_size = V, D
        self.add_adapter("general")
        self.set_adapter("general")

    def add_adapter(self, name):
        for b in self.blocks:
            b.up.add_adapter(name)
            b.down.add_adapter(name)

    def set_adapter(self, name):
        for b in self.blocks:
            b.up.active = name
            b.down.active = name

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=ids.device), 1
        )
        for b in self.blocks:
            x = b(x, mask)
        h = self.ln_f(x)
        return self.head(h), h

    @torch.no_grad()
    def generate(self, ids, n_new=16, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(ids[:, -self.SEQ_LEN :])
            if temperature <= 0:
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(
                    F.softmax(logits[:, -1] / temperature, -1), 1
                )
            ids = torch.cat([ids, nxt], dim=1)
        return ids

    @property
    def device(self):
        return next(self.parameters()).device


def make_backend(llm: str, **kw) -> LLMBackend:
    """'tiny' -> toy model; anything else -> HF hub id or local path."""
    return TinyBackend() if llm == "tiny" else HFBackend(llm, **kw)


def load_hf_backend_from_checkpoint(
    base_llm: str,
    adapter_dir: str | None,
    *,
    adapter_names: tuple[str, ...] = ("general",),
    device: str | None = None,
    load_in_4bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> HFBackend:
    """Load a frozen base LM + saved PEFT adapters (ensemble checkpoint llm/)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif resolved_device != "cpu":
        kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(base_llm, **kwargs)
    if not load_in_4bit and resolved_device != "cpu":
        base.to(resolved_device)
    for p in base.parameters():
        p.requires_grad_(False)

    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        adapters = set(adapter_names)
    else:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg, adapter_name="general")
        adapters = {"general"}

    backend = HFBackend.__new__(HFBackend)
    nn.Module.__init__(backend)
    backend.device_ = torch.device(resolved_device)
    backend.tokenizer = tokenizer
    backend.model = model
    backend._lora_cfg = None
    backend._adapters = adapters
    backend.vocab_size = model.config.vocab_size
    backend.hidden_size = model.config.hidden_size
    if adapter_names:
        backend.set_adapter(adapter_names[0])
    return backend


class TinyLLM(nn.Module):
    """Simpler toy LLM for the world-model track (no adapter bank)."""

    VOCAB, D, L, H, T = 1000, 128, 2, 4, 32

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(self.VOCAB, self.D)
        self.pos = nn.Embedding(self.T * 4, self.D)
        layer = nn.TransformerEncoderLayer(
            self.D, self.H, 4 * self.D, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, self.L)
        self.head = nn.Linear(self.D, self.VOCAB, bias=False)
        self.vocab_size, self.hidden_size = self.VOCAB, self.D

    def forward(self, ids):
        Tn = ids.size(1)
        x = self.tok(ids) + self.pos(torch.arange(Tn, device=ids.device))
        mask = torch.triu(
            torch.full((Tn, Tn), float("-inf"), device=ids.device), 1
        )
        h = self.blocks(x, mask=mask)
        return self.head(h), h

    @torch.no_grad()
    def generate(self, ids, n_new=16, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(ids[:, -self.T :])
            nxt = torch.multinomial(
                F.softmax(logits[:, -1] / temperature, -1), 1
            )
            ids = torch.cat([ids, nxt], 1)
        return ids

    def trainable_parameters(self):
        return self.parameters()

    @property
    def device(self):
        return next(self.parameters()).device


class HFLLM(nn.Module):
    """Small HF model with single LoRA stack (world-model track)."""

    def __init__(self, path, lora_r=16):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        for p in base.parameters():
            p.requires_grad_(False)
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, cfg)
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size

    def forward(self, ids):
        out = self.model(
            input_ids=ids.to(self.device), output_hidden_states=True
        )
        return out.logits, out.hidden_states[-1]

    @torch.no_grad()
    def generate(self, ids, n_new=32, temperature=0.8):
        return self.model.generate(
            input_ids=ids.to(self.device),
            max_new_tokens=n_new,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.model.parameters()).device


def load_llm(spec: str):
    return TinyLLM() if spec == "tiny" else HFLLM(spec)
