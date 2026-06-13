import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from inference.config import ModelConfig


class LlamaCppBackend:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: Llama | None = None
        self._model_path: str | None = None

    def _resolve_model_path(self) -> str:
        if self._config.model_path:
            path = Path(self._config.model_path)
            if not path.exists():
                raise FileNotFoundError(f"MODEL_PATH does not exist: {self._config.model_path}")
            return str(path)

        if not self._config.model_repo or not self._config.model_file:
            raise ValueError(
                f"Preset {self._config.key!r} requires model_repo and model_file for llama_cpp"
            )

        cache_dir = os.environ.get("MODEL_CACHE_DIR")

        return hf_hub_download(
            repo_id=self._config.model_repo,
            filename=self._config.model_file,
            cache_dir=cache_dir,
        )

    def unload(self) -> None:
        self._model = None
        self._model_path = None

    def load(self) -> None:
        if self._model is not None:
            return

        self._model_path = self._resolve_model_path()
        gpu_layers = self._config.n_gpu_layers
        try:
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._config.n_ctx,
                n_gpu_layers=gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            if gpu_layers <= 0:
                raise
            print(
                f"[inference] llama.cpp GPU offload failed ({exc}); using CPU (n_gpu_layers=0)…",
                flush=True,
            )
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._config.n_ctx,
                n_gpu_layers=0,
                verbose=False,
            )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.load()
        assert self._model is not None

        result = self._model(
            prompt,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.load()
        assert self._model is not None

        result = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
        )
        return result["choices"][0]["message"]["content"].strip()
