import importlib

import pytest


@pytest.fixture
def model_loading_module(monkeypatch, tmp_path):
    presets = tmp_path / "models.yaml"
    presets.write_text(
        """
defaults:
  active_model: alpha
  allow_model_switch: true
models:
  alpha:
    label: Alpha
    backend: transformers
    model_id: openbmb/MiniCPM5-1B
  beta:
    label: Beta GGUF
    backend: llama_cpp
    model_repo: openbmb/MiniCPM-V-4.6-gguf
    model_file: MiniCPM-V-4.6-Q4_K_M.gguf
    multimodal: true
"""
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACTIVE_MODEL", raising=False)

    import inference.config as inference_config
    import gradio_space.model_loading as model_loading

    importlib.reload(inference_config)
    importlib.reload(model_loading)
    return model_loading


def test_runtime_model_key_override(model_loading_module):
    ml = model_loading_module
    assert ml.get_active_model_key() == "alpha"
    ml.set_runtime_model_key("beta")
    assert ml.get_active_model_key() == "beta"


def test_set_runtime_model_key_unknown_raises(model_loading_module):
    ml = model_loading_module
    with pytest.raises(KeyError):
        ml.set_runtime_model_key("missing")
