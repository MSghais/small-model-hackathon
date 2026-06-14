from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any

from researchmind.citations import Citation, clean_model_answer, format_context_block, format_references
from researchmind.config import get_config
from researchmind.extract import ExtractedDocument
from researchmind.ingest import IngestPipeline
from researchmind.retrieve import retrieve
from researchmind.scope import rag_scope_warning, resolve_retrieve_scope
from researchmind.scrape_pdf import extract_pdf
from researchmind.scrape_web import fetch_and_extract
from researchmind.search_urls import search_urls
from researchmind.store import MemRAGStore
from researchmind.url_suggest import suggest_urls as llm_suggest_urls

from agent.research_prompts import research_answer_system, research_answer_user


def get_store() -> MemRAGStore:
    return IngestPipeline().store


def tool_suggest_urls(topic: str, backend: Any) -> list[str]:
    return llm_suggest_urls(topic, backend)


def tool_scrape_web(url: str) -> ExtractedDocument:
    return fetch_and_extract(url)


def tool_scrape_pdf(path: Path) -> ExtractedDocument:
    return extract_pdf(path)


def tool_extract_and_index(
    doc: ExtractedDocument,
    *,
    session_id: str | None = None,
) -> tuple[str, bool]:
    pipeline = IngestPipeline()
    return pipeline.ingest_document(doc, session_id=session_id)


def tool_research_answer(
    question: str,
    backend: Any,
    *,
    skill_body: str,
    skill_path: Path,
    session_id: str | None = None,
    doc_ids: list[str] | None = None,
    trace: Any | None = None,
) -> tuple[str, list[Citation], str]:
    cfg = get_config()
    store = get_store()
    scope_session, scope_docs = resolve_retrieve_scope(session_id, doc_ids)
    retrieve_started = monotonic()
    chunks = retrieve(
        question,
        store,
        config=cfg,
        session_id=scope_session,
        doc_ids=scope_docs,
    )
    if trace is not None:
        trace.log_step(
            "retrieve",
            "Retrieve passages",
            duration_ms=int((monotonic() - retrieve_started) * 1000),
            chunks=len(chunks),
            session_id=scope_session or "",
            doc_ids=scope_docs or [],
        )
    if not chunks:
        hint = rag_scope_warning(session_id=session_id, doc_ids=doc_ids)
        return hint, [], ""

    context, citations = format_context_block(chunks)
    system = research_answer_system(skill_body, skill_path)
    user = research_answer_user(question, context)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    generate_started = monotonic()
    answer = clean_model_answer(
        backend.chat(messages, max_tokens=512, temperature=0.2)
    )
    if trace is not None:
        trace.log_step(
            "generate",
            "Generate cited answer",
            duration_ms=int((monotonic() - generate_started) * 1000),
            citations=len(citations),
        )
    refs = format_references(citations)
    if session_id:
        store.add_message(session_id, "user", question, [c.chunk_id for c in citations])
        store.add_message(session_id, "assistant", answer, [c.chunk_id for c in citations])

    return answer, citations, refs


def tool_search_urls(topic: str, *, n: int = 5, check_reachable: bool = True) -> list[str]:
    return search_urls(topic, n=n, check_reachable=check_reachable)
