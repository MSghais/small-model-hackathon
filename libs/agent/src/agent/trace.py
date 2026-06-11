from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class TraceRecorder:
    skill: str
    model: str
    user_input: dict[str, Any]
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    steps: list[dict[str, Any]] = field(default_factory=list)
    artifact: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def log_llm(self, prompt: str, output: str) -> None:
        self.steps.append(
            {
                "type": "llm",
                "prompt_hash": _prompt_hash(prompt),
                "output": output,
            }
        )

    def log_note(self, message: str, **details: Any) -> None:
        self.steps.append({"type": "note", "message": message, **details})

    def log_tool(self, name: str, arguments: dict[str, Any], result: str) -> None:
        self.steps.append(
            {
                "type": "tool",
                "name": name,
                "arguments": arguments,
                "result": result,
            }
        )

    def set_artifact(self, path: str) -> None:
        self.artifact = path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "skill": self.skill,
            "model": self.model,
            "input": self.user_input,
            "steps": self.steps,
            "artifact": self.artifact,
            "created_at": self.created_at,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, traces_dir: Path | None = None) -> Path:
        base = traces_dir or _default_traces_dir()
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{self.run_id}.json"
        path.write_text(self.to_json())
        return path


def _default_traces_dir() -> Path:
    env = __import__("os").environ.get("AGENT_TRACES_DIR")
    if env:
        return Path(env)
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "outputs" / "traces"
        if (base / "models.yaml").is_file() or (base / "pyproject.toml").is_file():
            return candidate
    return Path.cwd() / "outputs" / "traces"
