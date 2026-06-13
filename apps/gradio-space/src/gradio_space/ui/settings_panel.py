from __future__ import annotations

import gradio as gr

from echocoach.config import get_echo_coach_config
from gradio_space.model_loading import model_status, reload_model
from inference.config import get_app_config
from researchmind.config import get_config as get_research_config

_app_config = get_app_config()


def _voice_stack_summary() -> str:
    cfg = get_echo_coach_config()
    asr = cfg.get_asr()
    tts = cfg.get_tts()
    lines = [
        f"- **ASR:** {asr.label} (`{cfg.asr_preset}`)",
        f"- **TTS:** {tts.label} (`{cfg.tts_preset}`)",
        f"- **Coach model:** `{cfg.coach_model}`",
        f"- **Max recording:** {cfg.max_seconds}s",
    ]
    if cfg.presets_path:
        lines.append(f"- Voice presets: `{cfg.presets_path}`")
    return "\n".join(lines)


def _paths_summary() -> str:
    rm = get_research_config()
    lines = []
    if _app_config.presets_path:
        lines.append(f"- **Model presets:** `{_app_config.presets_path}`")
    else:
        lines.append("- **Model presets:** built-in defaults")
    lines.append(f"- **ResearchMind store:** `{rm.data_dir.resolve()}`")
    return "\n".join(lines)


def build_settings_panel() -> tuple[gr.Dropdown | None, gr.Markdown, gr.Button]:
    """Build settings accordion contents. Returns (model_dropdown or None, status_md, reload_btn)."""
    model_dropdown: gr.Dropdown | None = None

    if _app_config.allow_model_switch and len(_app_config.models) > 1:
        model_dropdown = gr.Dropdown(
            choices=_app_config.model_choices(),
            value=_app_config.active_model,
            label="Model preset",
        )
    else:
        active = _app_config.active
        gr.Markdown(
            f"**Active model:** `{active.key}` — {active.label}  \n"
            f"**Backend:** `{active.backend}`"
        )

    status_md = gr.Markdown(value=model_status(_app_config.active_model))
    gr.Markdown("#### Voice stack")
    gr.Markdown(_voice_stack_summary())
    with gr.Accordion("Paths & files", open=False):
        gr.Markdown(_paths_summary())

    reload_btn = gr.Button("Reload model", variant="secondary", size="sm")

    if model_dropdown is not None:
        model_dropdown.change(fn=model_status, inputs=model_dropdown, outputs=status_md)

    if model_dropdown is not None:
        reload_btn.click(fn=reload_model, inputs=[model_dropdown], outputs=status_md)
    else:
        reload_btn.click(
            fn=lambda: reload_model(_app_config.active_model),
            outputs=status_md,
        )

    return model_dropdown, status_md, reload_btn
