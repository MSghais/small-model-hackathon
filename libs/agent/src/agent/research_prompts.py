from __future__ import annotations

from pathlib import Path


def _load_reference(skill_path: Path, rel: str) -> str:
    ref = skill_path.parent / rel
    if ref.is_file():
        return ref.read_text(encoding="utf-8")
    return ""


def research_answer_system(skill_body: str, skill_path: Path) -> str:
    citation_ref = _load_reference(skill_path, "references/citation-format.md")
    parts = [
        "You are ResearchMind, a local research assistant.",
        "Answer ONLY from the provided context. Cite sources inline as [1], [2], etc.",
        skill_body,
    ]
    if citation_ref:
        parts.append(citation_ref)
    return "\n\n".join(parts)


def research_answer_user(question: str, context: str) -> str:
    return f"""Context:
{context}

Question: {question}

Answer with inline [n] citations. If context is insufficient, say so."""
