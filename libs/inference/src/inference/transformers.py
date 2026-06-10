from inference.config import ModelConfig


class TransformersBackend:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._config.model_id:
            raise ValueError(
                f"Preset {self._config.key!r} requires model_id for transformers backend"
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers backend requires torch and transformers. "
                "Install with: uv sync --all-packages"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_id,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=self._config.trust_remote_code,
        )
        if device == "cpu":
            self._model.to(device)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        max_new_tokens = max_tokens or self._config.max_tokens
        temp = self._config.temperature if temperature is None else temperature

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            do_sample=temp > 0,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.load()
        assert self._model is not None
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
                role = message["role"]
                content = message["content"]
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            prompt = "\n".join(parts)

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
