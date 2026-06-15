from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.models import QuizOutline, SlideOutline
from agent.tools.pptx import create_pptx
from agent.tools.quiz import create_quiz
from agent.tools.research_tools import (
    tool_extract_and_index,
    tool_research_answer,
    tool_scrape_pdf,
    tool_scrape_web,
    tool_search_urls,
    tool_suggest_urls,
)

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.register(
            "create_pptx",
            "Create a PowerPoint file from a validated SlideOutline",
            self._handle_create_pptx,
        )
        self.register(
            "create_quiz",
            "Create DOCX and HTML quiz exports from a validated QuizOutline",
            self._handle_create_quiz,
        )
        self.register(
            "suggest_urls",
            "Suggest research URLs for a topic using the local LLM",
            tool_suggest_urls,
        )
        self.register(
            "scrape_web",
            "Fetch and extract text from a web URL",
            tool_scrape_web,
        )
        self.register(
            "scrape_pdf",
            "Extract text from a PDF file path",
            tool_scrape_pdf,
        )
        self.register(
            "extract_and_index",
            "Chunk, embed, and index an ExtractedDocument into MemRAG",
            tool_extract_and_index,
        )
        self.register(
            "research_answer",
            "Answer a question with RAG citations from MemRAG",
            tool_research_answer,
        )
        self.register(
            "search_urls",
            "Web search for URLs on a topic (DuckDuckGo)",
            tool_search_urls,
        )

    def register(self, name: str, description: str, handler: Callable[..., Any]) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, handler=handler)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool {name!r}")
        return self._tools[name]

    def _handle_create_pptx(self, outline: SlideOutline, run_id: str | None = None) -> str:
        path = create_pptx(outline, run_id=run_id)
        return str(path)

    def _handle_create_quiz(
        self,
        outline: QuizOutline,
        run_id: str | None = None,
    ) -> dict[str, str]:
        from agent.tools.pptx import get_outputs_dir

        output_dir = get_outputs_dir() / (run_id or "quiz")
        paths = create_quiz(outline, output_dir, stem="quiz")
        return {fmt: str(path) for fmt, path in paths.items()}
