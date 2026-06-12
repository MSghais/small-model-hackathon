from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SlideSpec(BaseModel):
    title: str
    bullets: list[str] = Field(default_factory=list, min_length=1)
    speaker_note: str = ""


class SlideOutline(BaseModel):
    title: str
    slides: list[SlideSpec] = Field(min_length=1)


class EducationPptxInput(BaseModel):
    topic: str
    grade: str
    slide_count: int = Field(ge=3, le=8)
    source_mode: Literal["none", "web", "rag"] = "none"
    search_workflow: Literal["two_step", "auto"] = "two_step"
    urls: list[str] = Field(default_factory=list)
    files: list[Path] = Field(default_factory=list)
    session_id: str | None = None
    doc_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_title: str
    doc_uri: str
    excerpt: str


class ResearchIngestInput(BaseModel):
    topic: str = ""
    urls: list[str] = Field(default_factory=list)
    auto_search: bool = False
    session_id: str | None = None


class ResearchChatInput(BaseModel):
    question: str
    session_id: str
    doc_ids: list[str] = Field(default_factory=list)


class ResearchDiscoverResult(BaseModel):
    suggested_urls: list[str]
    session_id: str
    trace_path: str


class IngestFailure(BaseModel):
    url: str
    reason: str
    stage: str = "unknown"


class ResearchIngestResult(BaseModel):
    session_id: str
    ingested: list[str]
    skipped: list[str]
    doc_ids: list[str] = Field(default_factory=list)
    failures: list[IngestFailure] = Field(default_factory=list)
    doc_count: int
    chunk_count: int
    trace_path: str
    message: str


class ResearchChatResult(BaseModel):
    answer: str
    citations: list[Citation]
    references_markdown: str
    session_id: str
    trace_path: str
