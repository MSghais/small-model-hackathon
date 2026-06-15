from __future__ import annotations

from inference.config import ModelConfig
from inference.device_utils import (
    DevicePlan,
    clear_cuda_cache,
    is_cuda_oom,
    iter_inference_device_plans,
)


class TransformersBackend:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device_label: str | None = None
        self._active_plan: DevicePlan | None = None

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device_label = None
        self._active_plan = None
        clear_cuda_cache()

    def _torch_dtype(self, plan: DevicePlan):
        import torch

        if plan.torch_dtype_name == "float16":
            return torch.float16
        return torch.float32

    def _load_on_plan(self, plan: DevicePlan) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        torch_dtype = self._torch_dtype(plan)
        common_kwargs = {
            "trust_remote_code": self._config.trust_remote_code,
        }
        model_kwargs = {
            **common_kwargs,
            "dtype": torch_dtype,
            "device_map": plan.device_map,
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

        if self._config.adapter_path:
            import re
            from pathlib import Path

            from peft import PeftModel

            adapter = self._config.adapter_path
            adapter_dir = Path(adapter)
            if adapter_dir.is_dir():
                # Local adapter (e.g. pulled from the Modal Volume).
                adapter_src = str(adapter_dir)
            elif re.fullmatch(r"[\w.-]+/[\w.-]+", adapter):
                # Hugging Face Hub repo id (e.g. the Modal-published adapter) —
                # PeftModel fetches it remotely; no manual pull required.
                adapter_src = adapter
            else:
                raise FileNotFoundError(
                    f"LoRA adapter not found for preset {self._config.key!r}: "
                    f"{adapter} (expected a local dir or a Hub repo id 'org/name')"
                )
            self._model = PeftModel.from_pretrained(self._model, adapter_src)

        if plan.device == "cpu":
            assert self._model is not None
            self._model.to("cpu")

        self._active_plan = plan
        self._device_label = plan.label

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._config.model_id:
            raise ValueError(
                f"Preset {self._config.key!r} requires model_id for transformers backend"
            )

        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
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

        last_error: Exception | None = None
        for plan in iter_inference_device_plans():
            self.unload()
            try:
                self._load_on_plan(plan)
                print(
                    f"[inference] Loaded {self._config.model_id} on {plan.label}",
                    flush=True,
                )
                return
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    last_error = exc
                    print(
                        f"[inference] CUDA OOM loading on {plan.label}; trying next device…",
                        flush=True,
                    )
                    continue
                raise
            except Exception as exc:
                if plan.device.startswith("cuda") and is_cuda_oom(exc):
                    last_error = exc
                    print(
                        f"[inference] Failed on {plan.label} ({exc}); trying next device…",
                        flush=True,
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to load model {self._config.model_id!r} on any device")

    def _on_cpu(self) -> bool:
        return self._active_plan is not None and self._active_plan.device == "cpu"

    def _move_model_to_cpu(self) -> None:
        assert self._model is not None
        clear_cuda_cache()
        self._model = self._model.to("cpu")
        if self._active_plan and self._active_plan.torch_dtype_name == "float16":
            self._model = self._model.float()
        self._active_plan = DevicePlan("cpu", "float32", None, "cpu (CUDA OOM fallback)")
        self._device_label = self._active_plan.label
        clear_cuda_cache()
        print(f"[inference] Moved {self._config.model_id} to CPU after CUDA OOM", flush=True)

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

    def _run_with_oom_fallback(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        try:
            return self._generate_from_messages(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except RuntimeError as exc:
            if self._on_cpu() or not is_cuda_oom(exc):
                raise
            self._move_model_to_cpu()
            return self._generate_from_messages(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

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
        return self._run_with_oom_fallback(
            normalized,
            max_tokens=max_tokens,
            temperature=temperature,
        )
