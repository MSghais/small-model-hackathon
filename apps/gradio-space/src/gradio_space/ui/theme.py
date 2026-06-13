from __future__ import annotations

from pathlib import Path

import gradio as gr

_CSS_PATH = Path(__file__).resolve().parent / "styles.css"


def get_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        button_primary_background_fill="#e86c00",
        button_primary_background_fill_hover="#cf6000",
        button_primary_text_color="white",
        block_title_text_weight="600",
    )


def load_css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")
