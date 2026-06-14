from agent.models import EducationPptxInput, SlideOutline, SlideSpec
from agent.preview import outline_to_html, render_slide_images
from agent.prompts import fallback_outline, outline_looks_like_schema_echo, outline_max_tokens
from agent.runner import AgentRunner
from agent.tools.docx import create_docx, create_html_export
from agent.tools.pptx import create_pptx


def test_outline_max_tokens_scales_with_slide_count():
    assert outline_max_tokens(5) == 750
    assert outline_max_tokens(1) == 230
    assert outline_max_tokens(20) == 1024
    runner = AgentRunner()
    raw = (
        '{"title": "AI Agents", "slides": ['
        '{"title": "Intro", "bullets": ["What is an agent?"]},'
        '{"title": "Uses", "bullets": ["Automation"]}'
        "]}"
    )
    outline = runner._parse_outline(raw, expected_slides=5)
    assert len(outline.slides) == 5
    assert outline.title == "AI Agents"


def test_parse_outline_trims_when_model_returns_too_many():
    runner = AgentRunner()
    raw = (
        '{"title": "Topic", "slides": ['
        '{"title": "A", "bullets": ["a"]},'
        '{"title": "B", "bullets": ["b"]},'
        '{"title": "C", "bullets": ["c"]},'
        '{"title": "D", "bullets": ["d"]}'
        "]}"
    )
    outline = runner._parse_outline(raw, expected_slides=3)
    assert len(outline.slides) == 3


def test_extract_json_from_fenced_block():
    raw = '```json\n{"title": "T", "slides": [{"title": "S", "bullets": ["a"]}]}\n```'
    data = AgentRunner._extract_json(raw)
    assert data["title"] == "T"


def test_extract_json_ignores_trailing_text():
    raw = (
        '{"title": "AI Agents", "slides": [{"title": "Intro", "bullets": ["a"]}]}\n'
        "Here is a short explanation of the lesson outline."
    )
    data = AgentRunner._extract_json(raw)
    assert data["title"] == "AI Agents"


def test_extract_json_ignores_duplicate_object():
    first = '{"title": "First", "slides": [{"title": "A", "bullets": ["a"]}]}'
    second = '{"title": "Second", "slides": [{"title": "B", "bullets": ["b"]}]}'
    data = AgentRunner._extract_json(f"{first}\n{second}")
    assert data["title"] == "First"


def test_extract_json_empty_raises():
    import pytest

    with pytest.raises(ValueError, match="empty output"):
        AgentRunner._extract_json("   ")


def test_extract_json_after_thinking_block():
    raw = (
        "planning the lesson\n"
        '{"title": "Agents", "slides": [{"title": "Intro", "bullets": ["What is an agent?"]}]}'
    )
    from inference.response_clean import strip_thinking_blocks

    cleaned = strip_thinking_blocks(raw)
    data = AgentRunner._extract_json(cleaned)
    assert data["title"] == "Agents"


def test_parse_outline_or_error_empty():
    runner = AgentRunner()
    outline, err = runner._parse_outline_or_error("", 5, None)
    assert outline is None
    assert "empty" in err.lower()


def test_parse_outline_rejects_schema_echo():
    runner = AgentRunner()
    raw = (
        '{"title": "string — presentation title", "slides": ['
        '{"title": "string — slide heading", "bullets": ["string", "..."], '
        '"speaker_note": "string — one sentence for the teacher"}'
        "]}"
    )
    import pytest

    with pytest.raises(ValueError, match="schema placeholders"):
        runner._parse_outline(raw, expected_slides=5)


def test_outline_looks_like_schema_echo():
    echo = SlideOutline(
        title="string — presentation title",
        slides=[SlideSpec(title="string — slide heading", bullets=["string", "..."])],
    )
    assert outline_looks_like_schema_echo(echo) is True

    real = SlideOutline(
        title="Small model finetuning",
        slides=[SlideSpec(title="What is finetuning?", bullets=["Adapting a base model"])],
    )
    assert outline_looks_like_schema_echo(real) is False


def test_fallback_outline_slide_count():
    req = EducationPptxInput(topic="ai agent", grade="6", slide_count=5)
    outline = fallback_outline(req)
    assert len(outline.slides) == 5
    assert "ai agent" in outline.title.lower()


def test_create_pptx_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OUTPUTS_DIR", str(tmp_path))
    outline = SlideOutline(
        title="Photosynthesis",
        slides=[
            SlideSpec(title="What is it?", bullets=["Plants make food", "Uses sunlight"]),
            SlideSpec(title="Why it matters", bullets=["Oxygen", "Food chain"]),
        ],
    )
    path = create_pptx(outline, run_id="test")
    assert path.exists()
    assert path.suffix == ".pptx"


def test_create_docx_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OUTPUTS_DIR", str(tmp_path))
    outline = SlideOutline(
        title="Photosynthesis",
        slides=[SlideSpec(title="Intro", bullets=["Sunlight", "Chlorophyll"])],
    )
    path = create_docx(outline, run_id="test")
    assert path.exists()
    assert path.suffix == ".docx"


def test_outline_preview_and_images(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OUTPUTS_DIR", str(tmp_path))
    outline = SlideOutline(
        title="Water Cycle",
        slides=[SlideSpec(title="Evaporation", bullets=["Heat", "Vapor"])],
    )
    html = outline_to_html(outline)
    assert "Water Cycle" in html
    assert "Evaporation" in html
    images = render_slide_images(outline, run_id="prev")
    assert len(images) == 2
    assert all(p.exists() for p in images)


def test_create_html_export(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OUTPUTS_DIR", str(tmp_path))
    outline = SlideOutline(
        title="Fractions",
        slides=[SlideSpec(title="Parts", bullets=["Half", "Quarter"])],
    )
    path = create_html_export(outline, run_id="html")
    assert path.exists()
    assert "Fractions" in path.read_text()
