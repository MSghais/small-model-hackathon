from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    task: str
    tools: list[str]
    model_hints: list[str]
    flags: dict[str, Any]
    body: str
    path: Path


def _find_skills_root() -> Path | None:
    import os

    env_path = os.environ.get("SKILLS_DIR")
    if env_path:
        root = Path(env_path)
        if root.is_dir():
            return root.resolve()

    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "skills"
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _parse_skill_md(path: Path) -> Skill:
    text = path.read_text()
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError(f"{path}: expected YAML frontmatter delimited by ---")

    meta: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()

    raw_flags = meta.get("flags") or {}
    flags = {str(k): v for k, v in raw_flags.items()} if isinstance(raw_flags, dict) else {}

    return Skill(
        name=str(meta.get("name", path.parent.name)),
        description=str(meta.get("description", "")),
        task=str(meta.get("task", "")),
        tools=[str(t) for t in meta.get("tools", [])],
        model_hints=[str(m) for m in meta.get("model_hints", [])],
        flags=flags,
        body=body,
        path=path,
    )


class SkillRegistry:
    def __init__(self, skills_root: Path | None = None) -> None:
        self._root = skills_root or _find_skills_root()
        self._skills: dict[str, Skill] = {}
        if self._root is not None:
            self._load_all()

    @property
    def root(self) -> Path | None:
        return self._root

    def _load_all(self) -> None:
        assert self._root is not None
        for skill_md in sorted(self._root.glob("*/SKILL.md")):
            skill = _parse_skill_md(skill_md)
            self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            known = ", ".join(sorted(self._skills)) or "(none found)"
            raise KeyError(f"Unknown skill {name!r}. Known: {known}")
        return self._skills[name]

    def list_by_task(self, task: str) -> list[Skill]:
        return [s for s in self._skills.values() if s.task == task]

    def all(self) -> list[Skill]:
        return list(self._skills.values())
