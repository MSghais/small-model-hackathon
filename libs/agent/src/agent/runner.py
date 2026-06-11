from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from inference.base import InferenceBackend

from agent.models import EducationPptxInput, SlideOutline, SlideSpec
from agent.preview import outline_to_html, render_slide_images
from agent.prompts import (
    education_outline_repair,
    education_outline_system,
    education_outline_user,
    outline_to_markdown,
)
from agent.skills import SkillRegistry
from agent.tools.docx import create_docx, create_html_export
from agent.tools_registry import ToolRegistry
from agent.trace import TraceRecorder

EDUCATION_PPTX_SKILL = "education-pptx"


@dataclass
class AgentResult:
    markdown_preview: str
    html_preview: str
    preview_images: list[str]
    pptx_path: str
    docx_path: str
    html_export_path: str
    trace: TraceRecorder
    trace_path: str
    outline: SlideOutline


class AgentRunner:
    def __init__(
        self,
        skills: SkillRegistry | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self._skills = skills or SkillRegistry()
        self._tools = tools or ToolRegistry()

    def run_education_pptx(
        self,
        *,
        topic: str,
        grade: str,
        slide_count: int,
        model_key: str,
        backend: InferenceBackend,
    ) -> AgentResult:
        skill = self._skills.get(EDUCATION_PPTX_SKILL)
        req = EducationPptxInput(topic=topic.strip(), grade=grade, slide_count=slide_count)

        trace = TraceRecorder(
            skill=skill.name,
            model=model_key,
            user_input=req.model_dump(),
        )

        backend.load()
        outline = self._generate_outline(skill, req, backend, trace)
        tool = self._tools.get("create_pptx")
        pptx_path = tool.handler(outline, run_id=trace.run_id)
        trace.log_tool(
            "create_pptx",
            {"title": outline.title, "slide_count": len(outline.slides)},
            pptx_path,
        )

        docx_path = create_docx(outline, run_id=trace.run_id)
        trace.log_tool(
            "create_docx",
            {"title": outline.title, "slide_count": len(outline.slides)},
            str(docx_path),
        )

        html_export_path = create_html_export(outline, run_id=trace.run_id)
        trace.log_tool(
            "create_html_export",
            {"title": outline.title},
            str(html_export_path),
        )

        trace.set_artifact(pptx_path)

        slides_dicts = [s.model_dump() for s in outline.slides]
        markdown = outline_to_markdown(outline.title, slides_dicts)
        html_preview = outline_to_html(outline)
        preview_images = [str(p) for p in render_slide_images(outline, trace.run_id)]
        trace_path = trace.save()

        return AgentResult(
            markdown_preview=markdown,
            html_preview=html_preview,
            preview_images=preview_images,
            pptx_path=pptx_path,
            docx_path=str(docx_path),
            html_export_path=str(html_export_path),
            trace=trace,
            trace_path=str(trace_path),
            outline=outline,
        )

    def _generate_outline(
        self,
        skill: Any,
        req: EducationPptxInput,
        backend: InferenceBackend,
        trace: TraceRecorder,
    ) -> SlideOutline:
        system = education_outline_system(skill.body)
        user = education_outline_user(req)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = system + "\n\n" + user
        raw = backend.chat(messages, max_tokens=2048, temperature=0.3)
        trace.log_llm(prompt_text, raw)

        try:
            return self._parse_outline(raw, req.slide_count, trace)
        except (json.JSONDecodeError, ValueError) as first_error:
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": education_outline_repair(
                        raw, str(first_error), expected_slides=req.slide_count
                    ),
                },
            ]
            repair_prompt = education_outline_repair(
                raw, str(first_error), expected_slides=req.slide_count
            )
            repaired = backend.chat(repair_messages, max_tokens=2048, temperature=0.1)
            trace.log_llm(repair_prompt, repaired)
            return self._parse_outline(repaired, req.slide_count, trace)

    def _parse_outline(
        self,
        raw: str,
        expected_slides: int,
        trace: TraceRecorder | None = None,
    ) -> SlideOutline:
        data = self._sanitize_outline_data(self._extract_json(raw))
        outline = SlideOutline.model_validate(data)
        original_count = len(outline.slides)
        outline = self._normalize_slide_count(outline, expected_slides)
        if trace and original_count != expected_slides:
            trace.log_note(
                "Adjusted slide count to match request",
                requested=expected_slides,
                model_returned=original_count,
                final=len(outline.slides),
            )
        return outline

    @staticmethod
    def _sanitize_outline_data(data: dict[str, Any]) -> dict[str, Any]:
        title = str(data.get("title") or "Lesson").strip() or "Lesson"
        slides_in = data.get("slides") or []
        slides_out: list[dict[str, Any]] = []
        for index, slide in enumerate(slides_in):
            if not isinstance(slide, dict):
                continue
            slide_title = str(slide.get("title") or f"Slide {index + 1}").strip()
            bullets_raw = slide.get("bullets") or []
            if isinstance(bullets_raw, str):
                bullets_raw = [bullets_raw]
            bullets = [str(b).strip() for b in bullets_raw if str(b).strip()]
            if not bullets:
                bullets = ["Discuss this topic with the class"]
            slides_out.append(
                {
                    "title": slide_title or f"Slide {index + 1}",
                    "bullets": bullets,
                    "speaker_note": str(slide.get("speaker_note") or ""),
                }
            )
        if not slides_out:
            slides_out.append(
                {
                    "title": "Introduction",
                    "bullets": ["Overview of the topic", "Why it matters"],
                    "speaker_note": "",
                }
            )
        return {"title": title, "slides": slides_out}

    @staticmethod
    def _normalize_slide_count(outline: SlideOutline, expected: int) -> SlideOutline:
        slides = list(outline.slides)
        if len(slides) > expected:
            slides = slides[:expected]
        while len(slides) < expected:
            number = len(slides) + 1
            slides.append(
                SlideSpec(
                    title=f"More about {outline.title}",
                    bullets=[
                        "Key idea to expand in class",
                        "Question for students",
                    ],
                    speaker_note="Add details for this slide during the lesson.",
                )
            )
        return SlideOutline(title=outline.title, slides=slides)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)
