from __future__ import annotations

from pathlib import Path

import gradio as gr

_CSS_PATH = Path(__file__).resolve().parent / "styles.css"


def get_theme() -> gr.Theme:
    """Neutral base theme — accent color only on explicit primary CTAs via CSS."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        button_primary_background_fill="#374151",
        button_primary_background_fill_hover="#1f2937",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#f3f4f6",
        button_secondary_background_fill_hover="#e5e7eb",
        block_label_background_fill="transparent",
        block_label_text_color="#4b5563",
        block_label_text_weight="500",
        block_title_text_weight="600",
        block_title_text_color="#111827",
        input_background_fill="#ffffff",
        body_text_color="#374151",
        border_color_primary="#e5e7eb",
        checkbox_label_background_fill_selected="#f3f4f6",
        checkbox_label_text_color_selected="#111827",
        checkbox_label_border_color_selected="#9ca3af",
    )


def load_css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")
