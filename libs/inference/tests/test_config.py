from pathlib import Path

import pytest

from inference.config import load_app_config


def test_load_app_config_from_models_yaml(tmp_path, monkeypatch):
    presets = tmp_path / "models.yaml"
    presets.write_text(
        """
defaults:
  active_model: demo
  allow_model_switch: true
models:
  demo:
    label: Demo preset
    backend: llama_cpp
    model_repo: org/model-GGUF
    model_file: demo.gguf
"""
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACTIVE_MODEL", raising=False)

    config = load_app_config()

    assert config.active_model == "demo"
    assert config.allow_model_switch is True
    assert config.get_model("demo").model_repo == "org/model-GGUF"


def test_legacy_env_overrides_active_preset(tmp_path, monkeypatch):
    presets = tmp_path / "models.yaml"
    presets.write_text(
        """
defaults:
  active_model: demo
models:
  demo:
    label: Demo
    backend: llama_cpp
    model_repo: org/original
    model_file: original.gguf
"""
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_REPO", "org/override")
    monkeypatch.setenv("MODEL_FILE", "override.gguf")

    model = load_app_config().get_model("demo")

    assert model.model_repo == "org/override"
    assert model.model_file == "override.gguf"


def test_minicpm_v_gguf_preset_from_repo(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    models_yaml = repo_root / "models.yaml"
    if not models_yaml.is_file():
        pytest.skip("repo models.yaml not found")

    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("ACTIVE_MODEL", raising=False)
    monkeypatch.delenv("ALLOW_MODEL_SWITCH", raising=False)

    model = load_app_config().get_model("minicpm-v-4.6-gguf")

    assert model.backend == "llama_cpp"
    assert model.multimodal is True
    assert model.model_repo == "openbmb/MiniCPM-V-4.6-gguf"
    assert model.model_file == "MiniCPM-V-4.6-Q4_K_M.gguf"


def test_resolve_relative_model_path(tmp_path, monkeypatch):
    local_dir = tmp_path / "gemma_merged_model"
    local_dir.mkdir()
    presets = tmp_path / "models.yaml"
    presets.write_text(
        f"""
defaults:
  active_model: local
models:
  local:
    label: Local merged
    backend: transformers
    model_id: ./{local_dir.name}
"""
    )
    monkeypatch.chdir(tmp_path)

    model = load_app_config().get_model("local")

    assert model.model_id == str(local_dir.resolve())
