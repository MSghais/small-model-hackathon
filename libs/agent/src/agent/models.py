from __future__ import annotations

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
