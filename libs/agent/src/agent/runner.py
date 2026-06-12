from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.base import InferenceBackend
from researchmind.extract import extract_docx
from researchmind.ingest import IngestPipeline

from agent.models import (
    Citation,
    EducationPptxInput,
    ResearchChatInput,
    ResearchChatResult,
    ResearchDiscoverResult,
    ResearchIngestResult,
    SlideOutline,
    SlideSpec,
)
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
RESEARCH_MIND_SKILL = "research-mind"


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

    def _research_skill(self) -> Any:
        return self._skills.get(RESEARCH_MIND_SKILL)

    def _ensure_session(
        self,
        store: Any,
        session_id: str | None,
        topic: str = "",
    ) -> str:
        if session_id and store.get_session(session_id):
            return session_id
        return store.create_session(topic=topic).id

    def run_researchmind_discover(
        self,
        *,
        topic: str,
        auto_search: bool,
        session_id: str | None,
        model_key: str,
        backend: InferenceBackend,
    ) -> ResearchDiscoverResult:
        skill = self._research_skill()
        pipeline = IngestPipeline()
        store = pipeline.store
        sid = self._ensure_session(store, session_id, topic=topic)

        trace = TraceRecorder(
            skill=skill.name,
            model=model_key,
            user_input={"topic": topic, "auto_search": auto_search, "phase": "discover"},
        )
        backend.load()

        search_tool = self._tools.get("search_urls")
        urls = search_tool.handler(topic, n=8)
        trace.log_tool(
            "search_urls",
            {"topic": topic, "n": 8, "queries": "google+ddg"},
            json.dumps(urls),
        )
        if not urls:
            suggest_tool = self._tools.get("suggest_urls")
            from researchmind.url_validate import filter_valid_urls

            raw_llm = suggest_tool.handler(topic, backend)
            urls = filter_valid_urls(raw_llm, check_reachable=True, max_results=5)
            trace.log_tool("suggest_urls", {"topic": topic, "fallback": True}, json.dumps(urls))

        trace_path = str(trace.save())
        return ResearchDiscoverResult(
            suggested_urls=urls,
            session_id=sid,
            trace_path=trace_path,
        )

    def run_researchmind_ingest(
        self,
        *,
        topic: str | None,
        urls: list[str],
        files: list[Path],
        auto_search: bool,
        session_id: str | None,
        model_key: str,
        backend: InferenceBackend,
    ) -> ResearchIngestResult:
        skill = self._research_skill()
        pipeline = IngestPipeline()
        store = pipeline.store
        sid = self._ensure_session(store, session_id, topic=topic or "")

        trace = TraceRecorder(
            skill=skill.name,
            model=model_key,
            user_input={
                "topic": topic,
                "urls": urls,
                "files": [str(f) for f in files],
                "auto_search": auto_search,
                "session_id": sid,
            },
        )
        backend.load()

        targets = [u.strip() for u in urls if u.strip()]
        if auto_search and topic and not targets and not files:
            discover = self.run_researchmind_discover(
                topic=topic,
                auto_search=True,
                session_id=sid,
                model_key=model_key,
                backend=backend,
            )
            targets = discover.suggested_urls

        from agent.models import IngestFailure

        ingested: list[str] = []
        skipped: list[str] = []
        failures: list[IngestFailure] = []

        scrape_web = self._tools.get("scrape_web")
        extract_index = self._tools.get("extract_and_index")

        from researchmind.url_validate import validate_url

        for url in targets:
            ok, reason, normalized = validate_url(url, check_reachable=False)
            if not ok:
                trace.log_note(f"Skipped invalid URL {url}", reason=reason, stage="validate")
                failures.append(IngestFailure(url=url, reason=reason, stage="validate"))
                continue
            try:
                doc = scrape_web.handler(normalized)
                if not (doc.text or "").strip():
                    msg = "empty content after scrape"
                    trace.log_note(f"Ingest failed for {url}", error=msg, stage="scrape")
                    failures.append(IngestFailure(url=url, reason=msg, stage="scrape"))
                    continue
                doc_id, is_new = extract_index.handler(doc, session_id=sid)
                trace.log_tool("scrape_web", {"url": url}, doc.title)
                trace.log_tool(
                    "extract_and_index",
                    {"uri": doc.uri},
                    f"{doc_id} new={is_new}",
                )
                (ingested if is_new else skipped).append(url)
            except Exception as exc:  # noqa: BLE001
                trace.log_note(f"Ingest failed for {url}", error=str(exc), stage="ingest")
                failures.append(IngestFailure(url=url, reason=str(exc), stage="ingest"))

        for file_path in files:
            path = Path(file_path)
            try:
                if path.suffix.lower() == ".pdf":
                    doc = self._tools.get("scrape_pdf").handler(path)
                elif path.suffix.lower() == ".docx":
                    doc = extract_docx(path)
                else:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    from researchmind.extract import ExtractedDocument

                    doc = ExtractedDocument(
                        source_type="file",
                        uri=str(path.resolve()),
                        title=path.stem,
                        text=text,
                    )
                doc_id, is_new = extract_index.handler(doc, session_id=sid)
                trace.log_tool("extract_and_index", {"file": str(path)}, f"{doc_id} new={is_new}")
                label = path.name
                (ingested if is_new else skipped).append(label)
            except Exception as exc:  # noqa: BLE001
                trace.log_note(f"Ingest failed for {path}", error=str(exc))
                skipped.append(path.name)

        doc_count = len(store.list_documents(session_id=sid))
        chunk_count = store.count_chunks()
        fail_n = len(failures)
        message = (
            f"Ingested {len(ingested)} source(s), skipped/duplicate {len(skipped)}, "
            f"failed {fail_n}. Session `{sid}` has {doc_count} document(s); "
            f"{chunk_count} total chunks."
        )
        trace.log_note(message, failures=[f.model_dump() for f in failures])
        trace_path = str(trace.save())

        return ResearchIngestResult(
            session_id=sid,
            ingested=ingested,
            skipped=skipped,
            failures=failures,
            doc_count=doc_count,
            chunk_count=chunk_count,
            trace_path=trace_path,
            message=message,
        )

    def run_researchmind_chat(
        self,
        *,
        question: str,
        session_id: str,
        model_key: str,
        backend: InferenceBackend,
        doc_ids: list[str] | None = None,
    ) -> ResearchChatResult:
        skill = self._research_skill()
        req = ResearchChatInput(
            question=question.strip(),
            session_id=session_id,
            doc_ids=doc_ids or [],
        )

        trace = TraceRecorder(
            skill=skill.name,
            model=model_key,
            user_input=req.model_dump(),
        )
        backend.load()

        answer_tool = self._tools.get("research_answer")
        raw_answer, citations, refs = answer_tool.handler(
            req.question,
            backend,
            skill_body=skill.body,
            skill_path=skill.path,
            session_id=req.session_id,
            doc_ids=req.doc_ids or None,
        )
        trace.log_llm(req.question, raw_answer)
        trace.log_note(
            "citations",
            count=len(citations),
            session_id=req.session_id,
            doc_ids=req.doc_ids,
        )

        full_answer = raw_answer
        if refs:
            full_answer = f"{raw_answer}\n\n{refs}"

        trace_path = str(trace.save())
        pydantic_citations = [
            Citation(
                index=c.index,
                chunk_id=c.chunk_id,
                doc_title=c.doc_title,
                doc_uri=c.doc_uri,
                excerpt=c.excerpt,
            )
            for c in citations
        ]

        return ResearchChatResult(
            answer=full_answer,
            citations=pydantic_citations,
            references_markdown=refs,
            session_id=req.session_id,
            trace_path=trace_path,
        )
