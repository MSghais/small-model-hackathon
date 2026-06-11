from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.models import SlideOutline
from agent.tools.pptx import create_pptx


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

    def register(self, name: str, description: str, handler: Callable[..., Any]) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, handler=handler)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool {name!r}")
        return self._tools[name]

    def _handle_create_pptx(self, outline: SlideOutline, run_id: str | None = None) -> str:
        path = create_pptx(outline, run_id=run_id)
        return str(path)
