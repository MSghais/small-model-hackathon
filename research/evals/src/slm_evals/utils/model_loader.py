"""
utils/model_loader.py
─────────────────────
Load a local (or HF Hub) model via HuggingFace Transformers.
Returns a model_bundle dict that every benchmark consumes.
"""

from __future__ import annotations
import torch
from pathlib import Path
from typing import Any


DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(
    model_path: str,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """
    Load model + tokenizer from a local path or HF Hub ID.

    Returns
    -------
    model_bundle : dict with keys
        model         – the loaded AutoModelForCausalLM
        tokenizer     – the matching AutoTokenizer
        device        – resolved torch device string
        dtype         – resolved torch dtype
        param_count   – float (billions)
        model_path    – original path string
        generate_fn   – convenience callable (prompt → str)
    """
    # ── lazy imports so the module is importable without torch installed ──────
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as e:
        raise ImportError(
            "transformers is required: pip install transformers accelerate"
        ) from e

    model_path = str(model_path)

    # ── Quantization config ───────────────────────────────────────────────────
    quant_cfg = None
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

    if dtype == "int4":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = None
    elif dtype == "int8":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )
    model.eval()

    # ── Parameter count ───────────────────────────────────────────────────────
    param_count = sum(p.numel() for p in model.parameters()) / 1e9

    # ── Resolve actual device ─────────────────────────────────────────────────
    resolved_device = next(model.parameters()).device

    # ── Convenience generate function ─────────────────────────────────────────
    def generate_fn(
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_strings: list[str] | None = None,
    ) -> str:
        """Run inference and return the decoded completion (without prompt)."""
        inputs = tokenizer(prompt, return_tensors="pt").to(resolved_device)

        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)

        # Strip the input tokens from output
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return {
        "model":        model,
        "tokenizer":    tokenizer,
        "device":       str(resolved_device),
        "dtype":        dtype,
        "param_count":  param_count,
        "model_path":   model_path,
        "generate_fn":  generate_fn,
    }
