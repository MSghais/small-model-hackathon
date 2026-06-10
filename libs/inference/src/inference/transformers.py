from __future__ import annotations

from inference.config import ModelConfig


class TransformersBackend:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device_label: str | None = None

    def _resolve_device(self):
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.float16, "auto"
        return "cpu", torch.float32, None

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._config.model_id:
            raise ValueError(
                f"Preset {self._config.key!r} requires model_id for transformers backend"
            )

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers backend requires torch and transformers. "
                "Install with: uv sync --all-packages"
            ) from exc

        device, torch_dtype, device_map = self._resolve_device()
        self._device_label = (
            f"cuda ({torch.cuda.get_device_name(0)})"
            if device == "cuda"
            else "cpu"
        )

        common_kwargs = {
            "trust_remote_code": self._config.trust_remote_code,
        }
        model_kwargs = {
            **common_kwargs,
            "dtype": torch_dtype,
            "device_map": device_map,
        }

        if self._config.multimodal:
            self._processor = AutoProcessor.from_pretrained(
                self._config.model_id,
                **common_kwargs,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._config.model_id,
                **model_kwargs,
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_id,
                **common_kwargs,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_id,
                **model_kwargs,
            )

        if device == "cpu":
            self._model.to(device)

    @property
    def device_label(self) -> str:
        self.load()
        return self._device_label or "unknown"

    def _normalize_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, object]]:
        if not self._config.multimodal:
            return messages

        normalized: list[dict[str, object]] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": message["role"], "content": content})
        return normalized

    def _generate_from_messages(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.load()
        assert self._model is not None

        max_new_tokens = max_tokens or self._config.max_tokens
        temp = self._config.temperature if temperature is None else temperature

        if self._config.multimodal:
            assert self._processor is not None
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=temp > 0,
            )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            return self._processor.decode(generated, skip_special_tokens=True).strip()

        assert self._tokenizer is not None
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            parts = []
            for message in messages:
                role = str(message["role"])
                content = str(message["content"])
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            prompt = "\n".join(parts)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            do_sample=temp > 0,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        normalized = self._normalize_messages(messages)
        return self._generate_from_messages(
            normalized,
            max_tokens=max_tokens,
            temperature=temperature,
        )
