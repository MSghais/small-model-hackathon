"""Save and load JEPA ensemble checkpoints under models/ensemble/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ensemble.backends import HFBackend, TinyBackend, make_backend
from ensemble.jepa_ensemble import Ensemble

MANIFEST_FILE = "manifest.json"
AUX_FILE = "aux.pt"
STORE_FILE = "store.pt"
LLM_DIR = "llm"
TINY_LLM_FILE = "tiny_llm.pt"

CHECKPOINT_VERSION = 1


def _aux_state_dict(ens: Ensemble) -> dict[str, torch.Tensor]:
    return {
        "emb": ens.emb.state_dict(),
        "jepa": ens.jepa.state_dict(),
        "bridge": ens.bridge.state_dict(),
        "router": ens.router.state_dict(),
    }


def _store_payload(ens: Ensemble) -> dict[str, Any]:
    return {
        "keys": [k for k in ens.store.keys],
        "values": [v for v in ens.store.values],
    }


def save_checkpoint(
    ens: Ensemble,
    out_dir: str | Path,
    *,
    base_llm: str,
    training_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist ensemble (LLM adapters + emb + JEPA + bridge + router + store)."""
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    backend = "tiny" if isinstance(ens.llm, TinyBackend) else "hf"
    manifest: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "track": "jepa",
        "backend": backend,
        "base_llm": base_llm,
        "adapter_names": list(ens.adapter_names),
        "d_emb": ens.emb.d_emb,
        "d_jepa": ens.jepa.d_latent,
        "training": training_meta or {},
    }

    torch.save(_aux_state_dict(ens), root / AUX_FILE)
    store = _store_payload(ens)
    if store["keys"]:
        torch.save(store, root / STORE_FILE)

    if backend == "hf":
        llm_path = root / LLM_DIR
        llm_path.mkdir(exist_ok=True)
        ens.llm.model.save_pretrained(llm_path)
        ens.llm.tokenizer.save_pretrained(llm_path)
    else:
        torch.save(ens.llm.state_dict(), root / TINY_LLM_FILE)

    with open(root / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    return root


def is_ensemble_checkpoint(path: str | Path) -> bool:
    return (Path(path) / MANIFEST_FILE).is_file()


def load_checkpoint(
    ckpt_dir: str | Path,
    *,
    device: str | None = None,
    load_in_4bit: bool = False,
) -> Ensemble:
    """Restore a saved JEPA ensemble from models/ensemble/<name>/."""
    root = Path(ckpt_dir).resolve()
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Not an ensemble checkpoint (missing {MANIFEST_FILE}): {root}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    base_llm = manifest["base_llm"]
    backend = manifest.get("backend", "hf")
    adapter_names = tuple(manifest.get("adapter_names", ["general"]))
    d_emb = manifest.get("d_emb", 64)
    d_jepa = manifest.get("d_jepa", 64)

    if backend == "tiny":
        ens = Ensemble(
            llm="tiny",
            adapter_names=adapter_names,
            d_emb=d_emb,
            d_jepa=d_jepa,
        )
        tiny_state = torch.load(root / TINY_LLM_FILE, map_location="cpu", weights_only=True)
        ens.llm.load_state_dict(tiny_state)
    else:
        ens = _load_hf_ensemble(
            root,
            base_llm=base_llm,
            adapter_names=adapter_names,
            d_emb=d_emb,
            d_jepa=d_jepa,
            device=device,
            load_in_4bit=load_in_4bit,
        )

    aux = torch.load(root / AUX_FILE, map_location="cpu", weights_only=True)
    ens.emb.load_state_dict(aux["emb"])
    ens.jepa.load_state_dict(aux["jepa"])
    ens.bridge.load_state_dict(aux["bridge"])
    ens.router.load_state_dict(aux["router"])

    store_path = root / STORE_FILE
    if store_path.is_file():
        store = torch.load(store_path, map_location="cpu", weights_only=True)
        ens.store.keys = list(store["keys"])
        ens.store.values = list(store["values"])

    ens.eval()
    return ens


def _load_hf_ensemble(
    root: Path,
    *,
    base_llm: str,
    adapter_names: tuple[str, ...],
    d_emb: int,
    d_jepa: int,
    device: str | None,
    load_in_4bit: bool,
) -> Ensemble:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    llm_dir = root / LLM_DIR
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        llm_dir if llm_dir.is_dir() else base_llm
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {}
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

    if llm_dir.is_dir():
        model = PeftModel.from_pretrained(base, str(llm_dir), is_trainable=False)
    else:
        model = base

    ens = Ensemble(
        llm=base_llm,
        adapter_names=adapter_names,
        d_emb=d_emb,
        d_jepa=d_jepa,
        load_in_4bit=load_in_4bit,
        device=resolved_device,
    )
    ens.llm.model = model
    ens.llm.tokenizer = tokenizer
    for name in adapter_names:
        if name not in ens.llm._adapters:
            ens.llm.add_adapter(name)
    ens.llm.set_adapter(adapter_names[0])
    return ens
