import os

import gradio as gr

from inference.factory import get_backend

_backend = get_backend()
_model_ready = False
_load_error: str | None = None


def _ensure_model_loaded() -> str | None:
    global _model_ready, _load_error

    if _model_ready:
        return None

    if _load_error:
        return _load_error

    try:
        _backend.load()
        _model_ready = True
        return None
    except Exception as exc:  # noqa: BLE001 — surface model load failures in the UI
        _load_error = f"Failed to load model: {exc}"
        return _load_error


def chat(message: str, history: list) -> str:
    load_error = _ensure_model_loaded()
    if load_error:
        return load_error

    messages: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})
    return _backend.chat(messages)


def build_demo() -> gr.Blocks:
    model_repo = os.environ.get("MODEL_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF")
    model_file = os.environ.get("MODEL_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf")
    backend_name = os.environ.get("INFERENCE_BACKEND", "llama_cpp")

    with gr.Blocks(title="Small Model Hackathon") as demo:
        gr.Markdown(
            f"""
# Small Model Chat

Local inference via **{backend_name}**. Model loads on first message.

- **Repo:** `{model_repo}`
- **File:** `{model_file}`

Part of the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).
"""
        )
        gr.ChatInterface(
            fn=chat,
            examples=["Hello! What can you help me with?", "Explain llama.cpp in one sentence."],
        )

    return demo


demo = build_demo()


def main() -> None:
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
