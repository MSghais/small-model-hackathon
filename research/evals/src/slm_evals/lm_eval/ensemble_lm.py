"""Custom lm-eval backend for saved JEPA ensemble checkpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


@register_model("ensemble-lm")
class EnsembleLM(LM):
    """Evaluate ensemble checkpoints with full-stack generation and LLM-head scoring."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = "auto",
        dtype: str = "bfloat16",
        batch_size: int | str = 1,
        max_batch_size: int | None = 64,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from ensemble.checkpoint import load_checkpoint

        load_in_4bit = dtype == "int4"
        resolved_device = None if device in (None, "auto") else device
        self._ens = load_checkpoint(
            checkpoint_path,
            device=resolved_device,
            load_in_4bit=load_in_4bit,
        )
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device or "auto"
        self._batch_size = batch_size
        self._max_batch_size = max_batch_size
        self._trust_remote_code = trust_remote_code

        backend_name = type(self._ens.llm).__name__
        if backend_name == "TinyBackend":
            self._hf_lm = None
            eval_logger.warning(
                "ensemble-lm: tiny backend checkpoint — loglikelihood tasks "
                "are not supported; use generate_until tasks only."
            )
        else:
            from lm_eval.models.huggingface import HFLM

            self._hf_lm = HFLM(
                pretrained=self._ens.llm.model,
                tokenizer=self._ens.llm.tokenizer,
                device=self._device,
                dtype=dtype if dtype != "int4" else "auto",
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                trust_remote_code=trust_remote_code,
            )
            self._device = self._hf_lm.device

    @property
    def tokenizer_name(self) -> str:
        if self._hf_lm is not None:
            return self._hf_lm.tokenizer_name
        return f"ensemble:{self._checkpoint_path}"

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        if self._hf_lm is None:
            raise NotImplementedError(
                "loglikelihood is not supported for tiny ensemble checkpoints"
            )
        return self._hf_lm.loglikelihood(requests)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        if self._hf_lm is None:
            raise NotImplementedError(
                "loglikelihood_rolling is not supported for tiny ensemble checkpoints"
            )
        return self._hf_lm.loglikelihood_rolling(requests)

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        del disable_tqdm
        outputs: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            do_sample = gen_kwargs.get("do_sample", False)
            temperature = float(gen_kwargs.get("temperature", 0.0 if not do_sample else 1.0))

            text = self._ens.generate_text(
                context,
                max_new_tokens=int(max_gen_toks),
                temperature=temperature,
            )
            if until:
                for stop in until:
                    if stop and stop in text:
                        text = text.split(stop, 1)[0]
            outputs.append(text.strip())
        return outputs
