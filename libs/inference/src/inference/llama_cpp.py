import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"


class LlamaCppBackend:
    def __init__(self) -> None:
        self._model: Llama | None = None
        self._model_path: str | None = None

    def _resolve_model_path(self) -> str:
        model_path = os.environ.get("MODEL_PATH")
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
            return str(path)

        model_repo = os.environ.get("MODEL_REPO", DEFAULT_MODEL_REPO)
        model_file = os.environ.get("MODEL_FILE", DEFAULT_MODEL_FILE)
        cache_dir = os.environ.get("MODEL_CACHE_DIR")

        return hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            cache_dir=cache_dir,
        )

    def load(self) -> None:
        if self._model is not None:
            return

        self._model_path = self._resolve_model_path()
        n_ctx = int(os.environ.get("N_CTX", "4096"))
        n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "0"))

        self._model = Llama(
            model_path=self._model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.load()
        assert self._model is not None

        result = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.load()
        assert self._model is not None

        result = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"].strip()
