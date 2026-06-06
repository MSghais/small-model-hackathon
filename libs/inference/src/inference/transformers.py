import os

from inference.base import InferenceBackend


class TransformersBackend:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers backend requires optional deps. "
                "Install with: uv sync --package inference --extra transformers"
            ) from exc

        model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            self._model.to(device)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
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


# Satisfy static type checkers that expect InferenceBackend.
_: InferenceBackend = TransformersBackend()
