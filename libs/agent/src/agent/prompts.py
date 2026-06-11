from __future__ import annotations

import json

from agent.models import EducationPptxInput


def education_outline_system(skill_body: str) -> str:
    return f"""You are a lesson-planning assistant for teachers.
Follow the skill workflow below and output ONLY valid JSON (no markdown fences).

Skill workflow:
{skill_body}

JSON schema:
{{
  "title": "string — presentation title",
  "slides": [
    {{
      "title": "string — slide heading",
      "bullets": ["string", "..."],
      "speaker_note": "string — one sentence for the teacher"
    }}
  ]
}}

Rules:
- Use exactly the requested number of content slides (title slide is added separately by the tool).
- Bullets should be short, age-appropriate, and factual.
- speaker_note is optional but helpful for each slide.
"""


def education_outline_user(req: EducationPptxInput) -> str:
    return (
        f"Topic: {req.topic}\n"
        f"Grade level: {req.grade}\n"
        f"Number of content slides: {req.slide_count}\n"
        "Return JSON only."
    )


def education_outline_repair(invalid_output: str, error: str) -> str:
    return (
        "The previous response was invalid JSON or did not match the schema.\n"
        f"Validation error: {error}\n"
        f"Previous output:\n{invalid_output}\n\n"
        "Return corrected JSON only, no explanation."
    )


def outline_to_markdown(title: str, slides: list[dict]) -> str:
    lines = [f"# {title}", ""]
    for index, slide in enumerate(slides, start=1):
        lines.append(f"## Slide {index}: {slide.get('title', 'Untitled')}")
        for bullet in slide.get("bullets", []):
            lines.append(f"- {bullet}")
        note = slide.get("speaker_note", "")
        if note:
            lines.append(f"\n*Teacher note:* {note}")
        lines.append("")
    return "\n".join(lines).strip()


def outline_json_example(slide_count: int) -> str:
    example = {
        "title": "Example Lesson",
        "slides": [
            {
                "title": f"Slide {i}",
                "bullets": ["Key point A", "Key point B"],
                "speaker_note": "Brief teaching tip.",
            }
            for i in range(1, slide_count + 1)
        ],
    }
    return json.dumps(example, indent=2)
