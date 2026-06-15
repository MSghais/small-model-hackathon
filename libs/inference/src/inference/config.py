"""Central model preset and app configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

BackendName = Literal["llama_cpp", "transformers"]

DEFAULT_PRESET_KEY = "qwen3b-gguf"


@dataclass(frozen=True)
class ModelConfig:
    """Single model preset used by inference backends and the Gradio UI."""

    key: str
    label: str
    backend: BackendName
    model_repo: str | None = None
    model_file: str | None = None
    model_path: str | None = None
    model_id: str | None = None
    adapter_path: str | None = None
    trust_remote_code: bool = False
    multimodal: bool = False
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    max_tokens: int = 512
    temperature: float = 0.7

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.backend,
            self.model_repo,
            self.model_file,
            self.model_path,
            self.model_id,
            self.adapter_path,
            self.trust_remote_code,
            self.multimodal,
            self.n_ctx,
            self.n_gpu_layers,
        )

    def summary(self) -> str:
        if self.backend == "llama_cpp":
            source = self.model_path or f"{self.model_repo}/{self.model_file}"
            return f"{self.label} · llama.cpp · {source}"
        return f"{self.label} · transformers · {self.model_id}"

    def resolve_paths(self, base_dir: Path) -> ModelConfig:
        updates: dict[str, Any] = {}

        if self.model_path:
            path = Path(self.model_path)
            if not path.is_absolute():
                updates["model_path"] = str((base_dir / path).resolve())

        if self.model_id and self.model_id.startswith(("./", "../")):
            updates["model_id"] = str((base_dir / self.model_id).resolve())

        if self.adapter_path and self.adapter_path.startswith(("./", "../")):
            updates["adapter_path"] = str((base_dir / self.adapter_path).resolve())

        return replace(self, **updates) if updates else self


@dataclass(frozen=True)
class AppConfig:
    """Runtime app configuration for dev and Hugging Face Space."""

    active_model: str
    models: dict[str, ModelConfig]
    allow_model_switch: bool = True
    model_cache_dir: str | None = None
    presets_path: Path | None = None

    def get_model(self, key: str | None = None) -> ModelConfig:
        model_key = key or self.active_model
        if model_key not in self.models:
            known = ", ".join(sorted(self.models))
            raise KeyError(f"Unknown model preset {model_key!r}. Known presets: {known}")
        return self.models[model_key]

    @property
    def active(self) -> ModelConfig:
        return self.get_model(self.active_model)

    def model_choices(self) -> list[tuple[str, str]]:
        return [(model.label, model.key) for model in self.models.values()]


def _builtin_presets() -> dict[str, ModelConfig]:
    return {
        DEFAULT_PRESET_KEY: ModelConfig(
            key=DEFAULT_PRESET_KEY,
            label="Qwen 2.5 3B Instruct (GGUF)",
            backend="llama_cpp",
            model_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
            model_file="qwen2.5-3b-instruct-q4_k_m.gguf",
        ),
        "minicpm5-1b": ModelConfig(
            key="minicpm5-1b",
            label="MiniCPM5 1B (Transformers)",
            backend="transformers",
            model_id="openbmb/MiniCPM5-1B",
            trust_remote_code=True,
        ),
        "gemma-merged-local": ModelConfig(
            key="gemma-merged-local",
            label="Fine-tuned merged model (local)",
            backend="transformers",
            model_id="./gemma_merged_model",
        ),
    }


def _find_presets_path() -> Path | None:
    env_path = os.environ.get("MODEL_PRESETS_PATH")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path.resolve()

    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "models.yaml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _repo_root_for(presets_path: Path | None) -> Path:
    if presets_path is not None:
        return presets_path.parent
    app_root = os.environ.get("APP_ROOT")
    if app_root:
        return Path(app_root).resolve()
    return Path.cwd().resolve()


def _parse_model_entry(key: str, raw: dict[str, Any]) -> ModelConfig:
    backend = str(raw.get("backend", "llama_cpp")).lower()
    if backend not in ("llama_cpp", "transformers"):
        raise ValueError(f"Preset {key!r}: backend must be llama_cpp or transformers")

    return ModelConfig(
        key=key,
        label=str(raw.get("label", key)),
        backend=backend,  # type: ignore[arg-type]
        model_repo=raw.get("model_repo"),
        model_file=raw.get("model_file"),
        model_path=raw.get("model_path"),
        model_id=raw.get("model_id"),
        adapter_path=raw.get("adapter_path"),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        multimodal=bool(raw.get("multimodal", False)),
        n_ctx=int(raw.get("n_ctx", 4096)),
        n_gpu_layers=int(raw.get("n_gpu_layers", 0)),
        max_tokens=int(raw.get("max_tokens", 512)),
        temperature=float(raw.get("temperature", 0.7)),
    )


def _load_presets_from_yaml(path: Path) -> tuple[dict[str, Any], dict[str, ModelConfig]]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Loading models.yaml requires PyYAML. Install with: uv add --package inference pyyaml"
        ) from exc

    data = yaml.safe_load(path.read_text()) or {}
    defaults = data.get("defaults", {})
    raw_models = data.get("models", {})
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError(f"{path}: expected non-empty top-level 'models' mapping")

    models = {key: _parse_model_entry(key, value) for key, value in raw_models.items()}
    return defaults, models


def _apply_legacy_env_overrides(model: ModelConfig) -> ModelConfig:
    """Keep single-model .env workflow working alongside preset keys."""

    updates: dict[str, Any] = {}

    backend = os.environ.get("INFERENCE_BACKEND")
    if backend:
        updates["backend"] = backend.lower()

    for field, env_name in (
        ("model_repo", "MODEL_REPO"),
        ("model_file", "MODEL_FILE"),
        ("model_path", "MODEL_PATH"),
        ("model_id", "MODEL_ID"),
    ):
        value = os.environ.get(env_name)
        if value:
            updates[field] = value

    if os.environ.get("TRUST_REMOTE_CODE", "").lower() in {"1", "true", "yes"}:
        updates["trust_remote_code"] = True

    for field, env_name in (
        ("n_ctx", "N_CTX"),
        ("n_gpu_layers", "N_GPU_LAYERS"),
        ("max_tokens", "MAX_TOKENS"),
    ):
        value = os.environ.get(env_name)
        if value is not None and value != "":
            updates[field] = int(value)

    temperature = os.environ.get("TEMPERATURE")
    if temperature is not None and temperature != "":
        updates["temperature"] = float(temperature)

    return replace(model, **updates) if updates else model


def load_app_config() -> AppConfig:
    presets_path = _find_presets_path()
    repo_root = _repo_root_for(presets_path)

    if presets_path is None:
        defaults: dict[str, Any] = {}
        models = _builtin_presets()
    else:
        defaults, models = _load_presets_from_yaml(presets_path)

    active_model = os.environ.get("ACTIVE_MODEL") or defaults.get(
        "active_model", DEFAULT_PRESET_KEY
    )
    if active_model not in models:
        active_model = next(iter(models))

    allow_model_switch = os.environ.get("ALLOW_MODEL_SWITCH")
    if allow_model_switch is None:
        allow_switch = bool(defaults.get("allow_model_switch", True))
    else:
        allow_switch = allow_model_switch.lower() in {"1", "true", "yes"}

    cache_dir = os.environ.get("MODEL_CACHE_DIR") or defaults.get("model_cache_dir")

    resolved_models = {
        key: model.resolve_paths(repo_root) for key, model in models.items()
    }

    has_legacy_override = any(
        os.environ.get(name)
        for name in (
            "INFERENCE_BACKEND",
            "MODEL_REPO",
            "MODEL_FILE",
            "MODEL_PATH",
            "MODEL_ID",
            "N_CTX",
            "N_GPU_LAYERS",
        )
    )
    if has_legacy_override:
        resolved_models[active_model] = _apply_legacy_env_overrides(
            resolved_models[active_model]
        )

    return AppConfig(
        active_model=active_model,
        models=resolved_models,
        allow_model_switch=allow_switch,
        model_cache_dir=cache_dir,
        presets_path=presets_path,
    )


_app_config: AppConfig | None = None


def get_app_config(reload: bool = False) -> AppConfig:
    global _app_config
    if _app_config is None or reload:
        _app_config = load_app_config()
    return _app_config


def get_model_config(key: str | None = None) -> ModelConfig:
    config = get_app_config()
    model_key = key or config.active_model
    return config.get_model(model_key)
