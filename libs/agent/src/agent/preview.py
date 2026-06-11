from __future__ import annotations

import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agent.models import SlideOutline
from agent.tools.pptx import _outputs_dir, _safe_filename


def outline_to_html(outline: SlideOutline) -> str:
    """Render slide-like cards for in-browser preview."""
    slides_html: list[str] = []
    slides_html.append(
        _slide_card_html(
            title=outline.title,
            subtitle="Lesson slides",
            bullets=[],
            speaker_note="",
            index=0,
            is_title=True,
        )
    )
    for index, slide in enumerate(outline.slides, start=1):
        slides_html.append(
            _slide_card_html(
                title=slide.title,
                subtitle="",
                bullets=slide.bullets,
                speaker_note=slide.speaker_note,
                index=index,
                is_title=False,
            )
        )

    return f"""
<div class="lesson-deck">
  <style>
    .lesson-deck {{
      font-family: Georgia, "Iowan Old Style", serif;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 960px;
    }}
    .lesson-slide {{
      border: 2px solid #5a3a22;
      border-radius: 12px;
      background: linear-gradient(180deg, #fbf6e8 0%, #f6efe1 100%);
      box-shadow: 0 4px 0 rgba(58,37,22,0.12);
      padding: 24px 28px;
      min-height: 180px;
    }}
    .lesson-slide.title-slide {{
      background: linear-gradient(135deg, #3a2516 0%, #5a3a22 100%);
      color: #f6efe1;
      min-height: 220px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}
    .lesson-slide .slide-index {{
      font-size: 11px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #8a4a2b;
      margin-bottom: 8px;
      font-weight: 700;
    }}
    .lesson-slide.title-slide .slide-index {{
      color: #e6a85c;
    }}
    .lesson-slide h3 {{
      margin: 0 0 12px 0;
      font-size: 1.5rem;
      line-height: 1.2;
    }}
    .lesson-slide .subtitle {{
      margin: 0;
      opacity: 0.85;
      font-style: italic;
    }}
    .lesson-slide ul {{
      margin: 0;
      padding-left: 1.25rem;
    }}
    .lesson-slide li {{
      margin-bottom: 6px;
      line-height: 1.45;
    }}
    .lesson-slide .speaker-note {{
      margin-top: 14px;
      padding-top: 10px;
      border-top: 1px dashed #8a6a48;
      font-size: 0.9rem;
      color: #5a3a22;
      font-style: italic;
    }}
  </style>
  {''.join(slides_html)}
</div>
"""


def _slide_card_html(
    *,
    title: str,
    subtitle: str,
    bullets: list[str],
    speaker_note: str,
    index: int,
    is_title: bool,
) -> str:
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    klass = "lesson-slide title-slide" if is_title else "lesson-slide"
    label = "Title" if is_title else f"Slide {index}"

    bullets_html = ""
    if bullets:
        items = "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
        bullets_html = f"<ul>{items}</ul>"

    note_html = ""
    if speaker_note:
        note_html = f'<div class="speaker-note">Teacher note: {html.escape(speaker_note)}</div>'

    subtitle_html = f'<p class="subtitle">{safe_subtitle}</p>' if subtitle else ""

    return f"""
<article class="{klass}">
  <div class="slide-index">{label}</div>
  <h3>{safe_title}</h3>
  {subtitle_html}
  {bullets_html}
  {note_html}
</article>
"""


def render_slide_images(outline: SlideOutline, run_id: str) -> list[Path]:
    """Render PNG thumbnails for gr.Gallery preview."""
    out_dir = _outputs_dir() / f"preview_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    paths: list[Path] = []

    title_path = out_dir / "00_title.png"
    _draw_slide_image(
        title_path,
        width,
        height,
        title=outline.title,
        subtitle="Generated lesson slides",
        bullets=[],
        is_title=True,
    )
    paths.append(title_path)

    for index, slide in enumerate(outline.slides, start=1):
        path = out_dir / f"{index:02d}_{_safe_filename(slide.title)}.png"
        _draw_slide_image(
            path,
            width,
            height,
            title=slide.title,
            subtitle="",
            bullets=slide.bullets,
            is_title=False,
        )
        paths.append(path)

    return paths


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_slide_image(
    path: Path,
    width: int,
    height: int,
    *,
    title: str,
    subtitle: str,
    bullets: list[str],
    is_title: bool,
) -> None:
    if is_title:
        bg = (58, 37, 22)
        fg = (246, 239, 225)
        accent = (230, 168, 92)
    else:
        bg = (251, 246, 232)
        fg = (42, 33, 24)
        accent = (138, 106, 72)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    margin = 80
    title_font = _load_font(56 if is_title else 44, bold=True)
    body_font = _load_font(30)
    small_font = _load_font(24)

    y = margin
    if is_title:
        draw.text((margin, height // 2 - 80), _wrap_text(title, 28), fill=fg, font=title_font)
        if subtitle:
            draw.text((margin, height // 2 + 40), subtitle, fill=accent, font=small_font)
    else:
        draw.text((margin, y), _wrap_text(title, 32), fill=fg, font=title_font)
        y += 90
        for bullet in bullets:
            line = _wrap_text(f"• {bullet}", 48)
            for part in line.split("\n"):
                draw.text((margin + 10, y), part, fill=fg, font=body_font)
                y += 42
            y += 8

    draw.rectangle([(0, 0), (width, 8)], fill=accent)
    image.save(path)


def _wrap_text(text: str, max_chars: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines) if lines else text
