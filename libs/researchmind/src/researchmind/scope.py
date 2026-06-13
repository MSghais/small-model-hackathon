"""Shared RAG retrieval scope rules for sessions, documents, and corpus."""

from __future__ import annotations


def resolve_retrieve_scope(
    session_id: str | None,
    doc_ids: list[str] | None,
) -> tuple[str | None, list[str] | None]:
    """Return (session_id, doc_ids) arguments for ``retrieve``.

    When explicit document IDs are provided, search those documents across the
    store. Otherwise scope to the session, or the entire corpus when neither
    session nor documents are set.
    """
    if doc_ids:
        return None, list(doc_ids)
    if session_id:
        return session_id, None
    return None, None


def rag_scope_warning(
    *,
    session_id: str | None,
    doc_ids: list[str] | None,
) -> str:
    if doc_ids:
        return "No passages in selected documents for this question."
    if session_id:
        return "No indexed sources in this session yet."
    return "No indexed sources in the corpus yet."


def retrieval_query(
    question: str,
    *,
    topic: str | None = None,
) -> str:
    """Build a retrieval query from the user question and optional focus topic."""
    question = question.strip()
    topic = (topic or "").strip()
    if not topic:
        return question
    if topic.lower() in question.lower():
        return question
    return f"{topic}: {question}"
