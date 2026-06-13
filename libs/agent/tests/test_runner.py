from agent.models import SlideOutline, SlideSpec
from agent.preview import outline_to_html, render_slide_images
from agent.prompts import outline_max_tokens
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
