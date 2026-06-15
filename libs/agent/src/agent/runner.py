from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Literal

from inference.base import InferenceBackend
from inference.response_clean import strip_thinking_blocks
from researchmind.citations import format_context_block
from researchmind.extract import extract_docx
from researchmind.ingest import IngestPipeline
from researchmind.retrieve import retrieve

from agent.models import (
    Citation,
    EducationPptxInput,
    QuizMakerInput,
    QuizOutline,
    QuizQuestion,
    ResearchChatInput,
    ResearchChatResult,
    ResearchDiscoverResult,
    ResearchIngestResult,
    SlideOutline,
    SlideSpec,
)
from agent.preview import outline_to_html, render_slide_images
from agent.progress import QuizGenerationProgress, SlideGenerationProgress
from agent.prompts import (
    education_outline_repair,
    education_outline_retry_user,
    education_outline_system,
    education_outline_user,
    fallback_outline,
    fallback_quiz,
    outline_json_example,
    outline_looks_like_schema_echo,
    outline_max_tokens,
    outline_to_markdown,
    quiz_json_example,
    quiz_max_tokens,
    quiz_outline_repair,
    quiz_outline_retry_user,
    quiz_outline_system,
    quiz_outline_user,
    quiz_to_markdown,
)
from agent.skills import SkillRegistry
from agent.tools.docx import create_docx, create_html_export
from agent.tools_registry import ToolRegistry
from agent.trace import TraceRecorder

EDUCATION_PPTX_SKILL = "education-pptx"
QUIZ_MAKER_SKILL = "quiz-maker"
RESEARCH_MIND_SKILL = "research-mind"

LessonSourceInput = EducationPptxInput | QuizMakerInput


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
    source_summary: str = ""


@dataclass
class QuizAgentResult:
    markdown_preview: str
    html_preview: str
    docx_path: str
    html_export_path: str
    trace: TraceRecorder
    trace_path: str
    outline: QuizOutline
    source_summary: str = ""


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
        source_mode: Literal["none", "web", "rag"] = "none",
        search_workflow: Literal["two_step", "auto"] = "two_step",
        urls: list[str] | None = None,
        files: list[Path] | None = None,
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        conversation_context: str = "",
        progress: SlideGenerationProgress | None = None,
        skip_preview_images: bool = False,
    ) -> AgentResult:
        result: AgentResult | None = None
        for item in self.iter_education_pptx(
            topic=topic,
            grade=grade,
            slide_count=slide_count,
            model_key=model_key,
            backend=backend,
            source_mode=source_mode,
            search_workflow=search_workflow,
            urls=urls,
            files=files,
            session_id=session_id,
            doc_ids=doc_ids,
            conversation_context=conversation_context,
            progress=progress,
            skip_preview_images=skip_preview_images,
        ):
            if isinstance(item, AgentResult):
                result = item
        if result is None:
            raise RuntimeError("Slide generation did not return a result")
        return result

    def iter_education_pptx(
        self,
        *,
        topic: str,
        grade: str,
        slide_count: int,
        model_key: str,
        backend: InferenceBackend,
        source_mode: Literal["none", "web", "rag"] = "none",
        search_workflow: Literal["two_step", "auto"] = "two_step",
        urls: list[str] | None = None,
        files: list[Path] | None = None,
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        conversation_context: str = "",
        progress: SlideGenerationProgress | None = None,
        skip_preview_images: bool = False,
    ) -> Iterator[SlideGenerationProgress | AgentResult]:
        skill = self._skills.get(EDUCATION_PPTX_SKILL)
        req = EducationPptxInput(
            topic=topic.strip(),
            grade=grade,
            slide_count=slide_count,
            source_mode=source_mode,
            search_workflow=search_workflow,
            urls=urls or [],
            files=files or [],
            session_id=session_id or None,
            doc_ids=doc_ids or [],
            conversation_context=(conversation_context or "").strip(),
        )

        trace = TraceRecorder(
            skill=skill.name,
            model=model_key,
            user_input=req.model_dump(mode="json"),
        )

        try:
            yield from self._iter_education_pptx_steps(
                req=req,
                skill=skill,
                model_key=model_key,
                backend=backend,
                trace=trace,
                progress=progress,
                skip_preview_images=skip_preview_images,
            )
        except Exception as exc:
            trace.log_note("Run failed", error=str(exc))
            try:
                trace.save()
            except OSError:
                pass
            raise

    def _iter_education_pptx_steps(
        self,
        *,
        req: EducationPptxInput,
        skill: Any,
        model_key: str,
        backend: InferenceBackend,
        trace: TraceRecorder,
        progress: SlideGenerationProgress | None,
        skip_preview_images: bool,
    ) -> Iterator[SlideGenerationProgress | AgentResult]:
        if req.conversation_context.strip():
            trace.log_note(
                "Conversation grounding",
                chars=len(req.conversation_context.strip()),
            )
        if progress is not None:
            progress.begin("load_model", "Load language model")
            yield progress
        load_started = monotonic()
        backend.load()
        load_ms = int((monotonic() - load_started) * 1000)
        trace.log_step("load_model", "Load language model", duration_ms=load_ms)

        if progress is not None:
            progress.begin(
                "gather_sources",
                "Gather lesson sources",
                detail=req.source_mode,
            )
            yield progress
        source_started = monotonic()
        source_context, source_summary, active_session = self._gather_lesson_source_context(
            req, backend, model_key, trace
        )
        source_ms = int((monotonic() - source_started) * 1000)
        trace.log_step(
            "gather_sources",
            "Gather lesson sources",
            duration_ms=source_ms,
            source_mode=req.source_mode,
        )
        if active_session:
            req = req.model_copy(update={"session_id": active_session})
        if progress is not None:
            yield progress

        if progress is not None:
            progress.begin(
                "generate_outline",
                "Generate slide outline",
                detail=f"{req.slide_count} slides · grade {req.grade}",
            )
            yield progress
        outline_started = monotonic()
        outline = self._generate_outline(
            skill, req, backend, trace, source_context=source_context, progress=progress
        )
        outline_ms = int((monotonic() - outline_started) * 1000)
        trace.log_step(
            "generate_outline",
            "Generate slide outline",
            duration_ms=outline_ms,
            slide_count=len(outline.slides),
        )
        for step in trace.steps:
            if step.get("type") == "note" and step.get("phase") == "outline_fallback":
                note = str(step.get("message") or "")
                source_summary = f"{source_summary}\n\n_{note}_".strip() if source_summary else f"_{note}_"

        if progress is not None:
            yield progress

        if progress is not None:
            progress.begin("create_exports", "Build PPTX, DOCX, and HTML exports")
            yield progress
        export_started = monotonic()
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
        export_ms = int((monotonic() - export_started) * 1000)
        trace.log_step(
            "create_exports",
            "Build PPTX, DOCX, and HTML exports",
            duration_ms=export_ms,
        )

        trace.set_artifact(pptx_path)

        slides_dicts = [s.model_dump() for s in outline.slides]
        markdown = outline_to_markdown(outline.title, slides_dicts)
        html_preview = outline_to_html(outline)

        if skip_preview_images:
            preview_images: list[str] = []
            trace.log_step(
                "render_previews",
                "Render slide thumbnails",
                duration_ms=0,
                detail="skipped (HTML preview only)",
            )
        else:
            if progress is not None:
                progress.begin(
                    "render_previews",
                    "Render slide thumbnails",
                    detail=f"{len(outline.slides) + 1} images",
                )
                yield progress
            preview_started = monotonic()
            preview_images = [str(p) for p in render_slide_images(outline, trace.run_id)]
            preview_ms = int((monotonic() - preview_started) * 1000)
            trace.log_step(
                "render_previews",
                "Render slide thumbnails",
                duration_ms=preview_ms,
                image_count=len(preview_images),
            )

        if progress is not None:
            progress.finish()
            yield progress

        trace_path = trace.save()

        yield AgentResult(
            markdown_preview=markdown,
            html_preview=html_preview,
            preview_images=preview_images,
            pptx_path=pptx_path,
            docx_path=str(docx_path),
            html_export_path=str(html_export_path),
            trace=trace,
            trace_path=str(trace_path),
            outline=outline,
            source_summary=source_summary,
        )

    def run_quiz_maker(
        self,
        *,
        topic: str,
        grade: str,
        question_count: int = 5,
        model_key: str,
        backend: InferenceBackend,
        source_mode: Literal["none", "web", "rag"] = "none",
        search_workflow: Literal["two_step", "auto"] = "two_step",
        urls: list[str] | None = None,
        files: list[Path] | None = None,
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        conversation_context: str = "",
        progress: QuizGenerationProgress | None = None,
    ) -> QuizAgentResult:
        result: QuizAgentResult | None = None
        for item in self.iter_quiz_maker(
            topic=topic,
            grade=grade,
            question_count=question_count,
            model_key=model_key,
            backend=backend,
            source_mode=source_mode,
            search_workflow=search_workflow,
            urls=urls,
            files=files,
            session_id=session_id,
            doc_ids=doc_ids,
            conversation_context=conversation_context,
            progress=progress,
        ):
            if isinstance(item, QuizAgentResult):
                result = item
        if result is None:
            raise RuntimeError("Quiz generation did not return a result")
        return result

    def iter_quiz_maker(
        self,
        *,
        topic: str,
        grade: str,
        question_count: int = 5,
        model_key: str,
        backend: InferenceBackend,
        source_mode: Literal["none", "web", "rag"] = "none",
        search_workflow: Literal["two_step", "auto"] = "two_step",
        urls: list[str] | None = None,
        files: list[Path] | None = None,
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        conversation_context: str = "",
        progress: QuizGenerationProgress | None = None,
    ) -> Iterator[QuizGenerationProgress | QuizAgentResult]:
        skill = self._skills.get(QUIZ_MAKER_SKILL)
        req = QuizMakerInput(
            topic=topic.strip(),
            grade=grade,
            question_count=question_count,
            source_mode=source_mode,
            search_workflow=search_workflow,
            urls=urls or [],
            files=files or [],
            session_id=session_id,
            doc_ids=doc_ids or [],
            conversation_context=(conversation_context or "").strip(),
        )

        trace = TraceRecorder(
            skill=skill.name,
            model=model_key,
            user_input=req.model_dump(mode="json"),
        )

        try:
            yield from self._iter_quiz_maker_steps(
                req=req,
                skill=skill,
                model_key=model_key,
                backend=backend,
                trace=trace,
                progress=progress,
            )
        except Exception as exc:
            trace.log_note("Run failed", error=str(exc))
            try:
                trace.save()
            except OSError:
                pass
            raise

    def _iter_quiz_maker_steps(
        self,
        *,
        req: QuizMakerInput,
        skill: Any,
        model_key: str,
        backend: InferenceBackend,
        trace: TraceRecorder,
        progress: QuizGenerationProgress | None,
    ) -> Iterator[QuizGenerationProgress | QuizAgentResult]:
        if req.conversation_context.strip():
            trace.log_note(
                "Conversation grounding",
                chars=len(req.conversation_context.strip()),
            )
        if progress is not None:
            progress.begin("load_model", "Load language model")
            yield progress
        load_started = monotonic()
        backend.load()
        load_ms = int((monotonic() - load_started) * 1000)
        trace.log_step("load_model", "Load language model", duration_ms=load_ms)

        if progress is not None:
            progress.begin(
                "gather_sources",
                "Gather lesson sources",
                detail=req.source_mode,
            )
            yield progress
        source_started = monotonic()
        source_context, source_summary, active_session = self._gather_lesson_source_context(
            req, backend, model_key, trace
        )
        source_ms = int((monotonic() - source_started) * 1000)
        trace.log_step(
            "gather_sources",
            "Gather lesson sources",
            duration_ms=source_ms,
            source_mode=req.source_mode,
        )
        if active_session:
            req = req.model_copy(update={"session_id": active_session})
        if progress is not None:
            yield progress

        if progress is not None:
            progress.begin(
                "generate_outline",
                "Generate quiz outline",
                detail=f"{req.question_count} questions · grade {req.grade}",
            )
            yield progress
        outline_started = monotonic()
        outline = self._generate_quiz_outline(
            skill, req, backend, trace, source_context=source_context, progress=progress
        )
        outline_ms = int((monotonic() - outline_started) * 1000)
        trace.log_step(
            "generate_outline",
            "Generate quiz outline",
            duration_ms=outline_ms,
            question_count=len(outline.questions),
        )
        for step in trace.steps:
            if step.get("type") == "note" and step.get("phase") == "outline_fallback":
                note = str(step.get("message") or "")
                source_summary = f"{source_summary}\n\n_{note}_".strip() if source_summary else f"_{note}_"

        if progress is not None:
            yield progress

        if progress is not None:
            progress.begin("create_exports", "Build DOCX and HTML quiz exports")
            yield progress
        export_started = monotonic()
        tool = self._tools.get("create_quiz")
        export_paths = tool.handler(outline, run_id=trace.run_id)
        trace.log_tool(
            "create_quiz",
            {"title": outline.title, "question_count": len(outline.questions)},
            json.dumps(export_paths),
        )
        docx_path = export_paths["docx"]
        html_export_path = export_paths["html"]
        export_ms = int((monotonic() - export_started) * 1000)
        trace.log_step(
            "create_exports",
            "Build DOCX and HTML quiz exports",
            duration_ms=export_ms,
        )

        trace.set_artifact(docx_path)

        markdown = quiz_to_markdown(outline)
        html_preview_path = Path(html_export_path)
        html_preview = html_preview_path.read_text(encoding="utf-8")

        if progress is not None:
            progress.finish()
            yield progress

        trace_path = trace.save()

        yield QuizAgentResult(
            markdown_preview=markdown,
            html_preview=html_preview,
            docx_path=docx_path,
            html_export_path=html_export_path,
            trace=trace,
            trace_path=str(trace_path),
            outline=outline,
            source_summary=source_summary,
        )

    def _gather_lesson_source_context(
        self,
        req: LessonSourceInput,
        backend: InferenceBackend,
        model_key: str,
        trace: TraceRecorder,
    ) -> tuple[str, str, str | None]:
        if req.source_mode == "none":
            return "", "", None

        pipeline = IngestPipeline()
        store = pipeline.store
        session_id = req.session_id
        ingest_summary = ""
        ingest: ResearchIngestResult | None = None

        if req.source_mode == "web":
            if req.search_workflow == "two_step" and not req.urls and not req.files:
                raise ValueError(
                    "Two-step web search requires selected URLs, pasted URLs, or uploaded files. "
                    "Click **Discover sources** first, then select sources before generating."
                )
            auto_search = req.search_workflow == "auto"
            ingest = self.run_researchmind_ingest(
                topic=req.topic,
                urls=req.urls,
                files=req.files,
                auto_search=auto_search,
                session_id=session_id,
                model_key=model_key,
                backend=backend,
            )
            session_id = ingest.session_id
            ingest_summary = ingest.message
            trace.log_note(ingest.message, phase="lesson_ingest", session_id=session_id)
        elif req.source_mode == "rag":
            session_id = self._ensure_session(store, session_id, topic=req.topic)
            if req.urls or req.files:
                ingest = self.run_researchmind_ingest(
                    topic=req.topic,
                    urls=req.urls,
                    files=req.files,
                    auto_search=False,
                    session_id=session_id,
                    model_key=model_key,
                    backend=backend,
                )
                session_id = ingest.session_id
                ingest_summary = ingest.message
                trace.log_note(ingest.message, phase="lesson_ingest", session_id=session_id)

            doc_count = len(store.list_documents(session_id=session_id))
            resolved = self._lesson_doc_ids(store, session_id, req, ingest)
            if doc_count == 0 and not resolved:
                raise ValueError(
                    "RAG mode requires indexed sources. Select a ResearchMind session with "
                    "documents, or paste URLs / upload files on this tab."
                )

        scope_session, scope_docs = self._lesson_retrieve_scope(
            store, session_id, req, ingest
        )
        chunks = retrieve(
            req.topic,
            store,
            session_id=scope_session,
            doc_ids=scope_docs,
        )
        if not chunks:
            warning = (
                "No passages retrieved from indexed sources; outline uses model knowledge only."
            )
            trace.log_note(warning, session_id=session_id, doc_ids=req.doc_ids)
            summary = ingest_summary or warning
            if ingest_summary:
                summary = f"{ingest_summary}\n\n_{warning}_"
            return "", summary, session_id

        context, citations = format_context_block(chunks)
        trace.log_note(
            f"Retrieved {len(chunks)} passage(s) from {len(citations)} source(s)",
            phase="lesson_retrieve",
            passage_count=len(chunks),
            citation_count=len(citations),
            session_id=session_id,
            doc_ids=req.doc_ids,
        )
        retrieve_line = (
            f"Retrieved **{len(chunks)}** passage(s) from **{len(citations)}** source(s) "
            f"for outline grounding."
        )
        summary = f"{ingest_summary}\n\n{retrieve_line}".strip() if ingest_summary else retrieve_line
        return context, summary, session_id

    @staticmethod
    def _normalize_outline_llm_text(raw: str) -> str:
        return strip_thinking_blocks(raw)

    def _generate_outline(
        self,
        skill: Any,
        req: EducationPptxInput,
        backend: InferenceBackend,
        trace: TraceRecorder,
        *,
        source_context: str = "",
        progress: SlideGenerationProgress | None = None,
    ) -> SlideOutline:
        system = education_outline_system(skill.body)
        user = education_outline_user(req, source_context=source_context)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = system + "\n\n" + user
        token_budget = outline_max_tokens(req.slide_count)

        raw = self._normalize_outline_llm_text(
            backend.chat(messages, max_tokens=token_budget, temperature=0.0)
        )
        trace.log_llm(prompt_text, raw)

        if not raw:
            trace.log_note(
                "Empty outline response; retrying with JSON example",
                phase="outline_retry",
            )
            example = outline_json_example(req.slide_count)
            retry_user = education_outline_retry_user(req, example_json=example)
            retry_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": retry_user},
            ]
            retry_prompt = system + "\n\n" + retry_user
            raw = self._normalize_outline_llm_text(
                backend.chat(retry_messages, max_tokens=token_budget, temperature=0.0)
            )
            trace.log_llm(retry_prompt, raw)

        outline, parse_error = self._parse_outline_or_error(raw, req.slide_count, trace)
        if outline is not None:
            return outline

        if progress is not None:
            progress.begin(
                "repair_outline",
                "Repair outline JSON",
                detail=(parse_error or "invalid JSON")[:80],
            )
        repair_started = monotonic()
        repair_user = education_outline_repair(
            raw,
            parse_error or "invalid JSON",
            expected_slides=req.slide_count,
        )
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair_user},
        ]
        repaired = self._normalize_outline_llm_text(
            backend.chat(
                repair_messages,
                max_tokens=min(512, token_budget),
                temperature=0.0,
            )
        )
        trace.log_llm(repair_user, repaired)
        outline, repair_error = self._parse_outline_or_error(
            repaired, req.slide_count, trace
        )
        repair_ms = int((monotonic() - repair_started) * 1000)
        if outline is not None:
            trace.log_step(
                "repair_outline",
                "Repair outline JSON",
                duration_ms=repair_ms,
            )
            return outline

        trace.log_step(
            "repair_outline",
            "Repair outline JSON",
            duration_ms=repair_ms,
            error=repair_error or parse_error,
        )
        trace.log_note(
            "Model outline invalid after repair; using template slides.",
            phase="outline_fallback",
        )
        if progress is not None:
            progress.begin(
                "fallback_outline",
                "Use template outline",
                detail=(repair_error or parse_error or "invalid JSON")[:80],
            )
        return fallback_outline(req)

    def _generate_quiz_outline(
        self,
        skill: Any,
        req: QuizMakerInput,
        backend: InferenceBackend,
        trace: TraceRecorder,
        *,
        source_context: str = "",
        progress: QuizGenerationProgress | None = None,
    ) -> QuizOutline:
        system = quiz_outline_system(skill.body)
        user = quiz_outline_user(req, source_context=source_context)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = system + "\n\n" + user
        token_budget = quiz_max_tokens(req.question_count)

        raw = self._normalize_outline_llm_text(
            backend.chat(messages, max_tokens=token_budget, temperature=0.0)
        )
        trace.log_llm(prompt_text, raw)

        if not raw:
            trace.log_note(
                "Empty quiz outline response; retrying with JSON example",
                phase="outline_retry",
            )
            example = quiz_json_example(req.question_count)
            retry_user = quiz_outline_retry_user(req, example_json=example)
            retry_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": retry_user},
            ]
            retry_prompt = system + "\n\n" + retry_user
            raw = self._normalize_outline_llm_text(
                backend.chat(retry_messages, max_tokens=token_budget, temperature=0.0)
            )
            trace.log_llm(retry_prompt, raw)

        outline, parse_error = self._parse_quiz_outline_or_error(
            raw, req.question_count, trace
        )
        if outline is not None:
            return outline

        if progress is not None:
            progress.begin(
                "repair_outline",
                "Repair quiz JSON",
                detail=(parse_error or "invalid JSON")[:80],
            )
        repair_started = monotonic()
        repair_user = quiz_outline_repair(
            raw,
            parse_error or "invalid JSON",
            expected_questions=req.question_count,
        )
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair_user},
        ]
        repaired = self._normalize_outline_llm_text(
            backend.chat(
                repair_messages,
                max_tokens=min(768, token_budget),
                temperature=0.0,
            )
        )
        trace.log_llm(repair_user, repaired)
        outline, repair_error = self._parse_quiz_outline_or_error(
            repaired, req.question_count, trace
        )
        repair_ms = int((monotonic() - repair_started) * 1000)
        if outline is not None:
            trace.log_step(
                "repair_outline",
                "Repair quiz JSON",
                duration_ms=repair_ms,
            )
            return outline

        trace.log_step(
            "repair_outline",
            "Repair quiz JSON",
            duration_ms=repair_ms,
            error=repair_error or parse_error,
        )
        trace.log_note(
            "Model quiz outline invalid after repair; using template questions.",
            phase="outline_fallback",
        )
        if progress is not None:
            progress.begin(
                "fallback_outline",
                "Use template quiz",
                detail=(repair_error or parse_error or "invalid JSON")[:80],
            )
        return fallback_quiz(req)

    def _parse_quiz_outline_or_error(
        self,
        raw: str,
        expected_questions: int,
        trace: TraceRecorder | None,
    ) -> tuple[QuizOutline | None, str]:
        if not raw.strip():
            return None, "Model returned empty output (no JSON)"
        try:
            return self._parse_quiz_outline(raw, expected_questions, trace), ""
        except (json.JSONDecodeError, ValueError) as exc:
            return None, str(exc)

    def _parse_quiz_outline(
        self,
        raw: str,
        expected_questions: int,
        trace: TraceRecorder | None = None,
    ) -> QuizOutline:
        data = self._sanitize_quiz_data(self._extract_json(raw))
        outline = QuizOutline.model_validate(data)
        original_count = len(outline.questions)
        outline = self._normalize_question_count(outline, expected_questions)
        if trace and original_count != expected_questions:
            trace.log_note(
                "Adjusted question count to match request",
                requested=expected_questions,
                model_returned=original_count,
                final=len(outline.questions),
            )
        return outline

    @staticmethod
    def _sanitize_quiz_data(data: dict[str, Any]) -> dict[str, Any]:
        title = str(data.get("title") or "Quiz").strip() or "Quiz"
        instructions = str(data.get("instructions") or "").strip()
        questions_in = data.get("questions") or []
        questions_out: list[dict[str, Any]] = []
        for index, question in enumerate(questions_in):
            if not isinstance(question, dict):
                continue
            prompt = str(question.get("prompt") or f"Question {index + 1}?").strip()
            choices_raw = question.get("choices") or []
            if isinstance(choices_raw, str):
                choices_raw = [choices_raw]
            choices = [str(c).strip() for c in choices_raw if str(c).strip()]
            while len(choices) < 4:
                choices.append(f"Option {len(choices) + 1}")
            choices = choices[:4]
            correct_index = int(question.get("correct_index", 0))
            correct_index = max(0, min(3, correct_index))
            questions_out.append(
                {
                    "prompt": prompt or f"Question {index + 1}?",
                    "choices": choices,
                    "correct_index": correct_index,
                    "explanation": str(question.get("explanation") or ""),
                }
            )
        if not questions_out:
            questions_out.append(
                {
                    "prompt": "Sample question?",
                    "choices": ["Answer A", "Answer B", "Answer C", "Answer D"],
                    "correct_index": 0,
                    "explanation": "",
                }
            )
        return {"title": title, "instructions": instructions, "questions": questions_out}

    @staticmethod
    def _normalize_question_count(outline: QuizOutline, expected: int) -> QuizOutline:
        questions = list(outline.questions)
        if len(questions) > expected:
            questions = questions[:expected]
        while len(questions) < expected:
            number = len(questions) + 1
            questions.append(
                QuizQuestion(
                    prompt=f"Additional question {number} about {outline.title}?",
                    choices=["Correct", "Distractor A", "Distractor B", "Distractor C"],
                    correct_index=0,
                    explanation="",
                )
            )
        return outline.model_copy(update={"questions": questions})

    def _parse_outline_or_error(
        self,
        raw: str,
        expected_slides: int,
        trace: TraceRecorder | None,
    ) -> tuple[SlideOutline | None, str]:
        if not raw.strip():
            return None, "Model returned empty output (no JSON)"
        try:
            return self._parse_outline(raw, expected_slides, trace), ""
        except (json.JSONDecodeError, ValueError) as exc:
            return None, str(exc)

    def _parse_outline(
        self,
        raw: str,
        expected_slides: int,
        trace: TraceRecorder | None = None,
    ) -> SlideOutline:
        data = self._sanitize_outline_data(self._extract_json(raw))
        outline = SlideOutline.model_validate(data)
        if outline_looks_like_schema_echo(outline):
            raise ValueError(
                "Model echoed JSON schema placeholders instead of lesson content"
            )
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
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence:
            cleaned = fence.group(1).strip()

        if not cleaned:
            raise ValueError("Model returned empty output (no JSON)")

        start = cleaned.find("{")
        if start < 0:
            preview = cleaned[:120].replace("\n", " ")
            raise ValueError(f"Model response has no JSON object: {preview!r}")

        end = AgentRunner._matching_brace_end(cleaned, start)
        if end is not None:
            return json.loads(cleaned[start : end + 1])

        fallback_end = cleaned.rfind("}")
        if fallback_end > start:
            return json.loads(cleaned[start : fallback_end + 1])
        return json.loads(cleaned)

    @staticmethod
    def _matching_brace_end(text: str, start: int) -> int | None:
        """Return index of the closing brace that matches ``start`` (must be ``{``)."""
        if start >= len(text) or text[start] != "{":
            return None
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        return None

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

    @staticmethod
    def _lesson_doc_ids(
        store: Any,
        session_id: str | None,
        req: LessonSourceInput,
        ingest: ResearchIngestResult | None,
    ) -> list[str]:
        if req.doc_ids:
            return list(req.doc_ids)

        resolved: list[str] = []
        seen: set[str] = set()

        def add(doc_id: str | None) -> None:
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                resolved.append(doc_id)

        if ingest:
            for doc_id in ingest.doc_ids:
                add(doc_id)

        for doc in store.list_documents(session_id=session_id):
            add(doc.id)

        if ingest and not resolved:
            from researchmind.url_validate import validate_url

            for label in (*ingest.ingested, *ingest.skipped):
                ok, _, normalized = validate_url(label, check_reachable=False)
                if ok:
                    add(store.find_document_id_by_uri(normalized))
                add(store.find_document_id_by_uri(label))

        return resolved

    @staticmethod
    def _lesson_retrieve_scope(
        store: Any,
        session_id: str | None,
        req: LessonSourceInput,
        ingest: ResearchIngestResult | None,
    ) -> tuple[str | None, list[str] | None]:
        from researchmind.scope import resolve_retrieve_scope

        doc_ids = AgentRunner._lesson_doc_ids(store, session_id, req, ingest)
        return resolve_retrieve_scope(session_id, doc_ids or None)

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
        doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
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
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc_ids.append(doc_id)
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
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc_ids.append(doc_id)
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
            doc_ids=doc_ids,
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
        load_started = monotonic()
        backend.load()
        trace.log_step(
            "load_model",
            "Load language model",
            duration_ms=int((monotonic() - load_started) * 1000),
        )

        answer_tool = self._tools.get("research_answer")
        raw_answer, citations, refs = answer_tool.handler(
            req.question,
            backend,
            skill_body=skill.body,
            skill_path=skill.path,
            session_id=req.session_id,
            doc_ids=req.doc_ids or None,
            trace=trace,
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
