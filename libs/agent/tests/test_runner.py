from agent.models import SlideOutline, SlideSpec
from agent.runner import AgentRunner
from agent.tools.pptx import create_pptx


def test_extract_json_from_fenced_block():
    raw = '```json\n{"title": "T", "slides": [{"title": "S", "bullets": ["a"]}]}\n```'
    data = AgentRunner._extract_json(raw)
    assert data["title"] == "T"


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
