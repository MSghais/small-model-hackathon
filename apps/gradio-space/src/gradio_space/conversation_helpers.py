from __future__ import annotations

MAX_CONVERSATION_CHARS = 8000


def _strip_message_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("text", "message", "content"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _append_turn(lines: list[str], role: str, content: str) -> None:
    text = content.strip()
    if not text:
        return
    label = "User" if role == "user" else "Assistant"
    lines.append(f"{label}: {text}")


def _iter_turns(history: list, history_kind: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    kind = (history_kind or "gradio").strip().lower()

    if kind == "research":
        for msg in history or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            turns.append((role, _strip_message_text(msg.get("content"))))
        return turns

    for item in history or []:
        if isinstance(item, dict) and item.get("role"):
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            turns.append((role, _strip_message_text(item.get("content"))))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text = _strip_message_text(item[0])
            assistant_text = _strip_message_text(item[1])
            if user_text:
                turns.append(("user", user_text))
            if assistant_text:
                turns.append(("assistant", assistant_text))

    return turns


def format_conversation_context(
    history: list | None,
    history_kind: str = "gradio",
) -> tuple[str, str]:
    """Normalize chat history into transcript text and a derived topic."""
    turns = _iter_turns(history or [], history_kind)
    if not turns:
        return "", ""

    lines: list[str] = []
    derived_topic = ""
    for role, content in turns:
        if role == "user" and not derived_topic and content.strip():
            derived_topic = content.strip()[:200]
        _append_turn(lines, role, content)

    if not lines:
        return "", derived_topic

    full = "\n\n".join(lines)
    if len(full) <= MAX_CONVERSATION_CHARS:
        return full, derived_topic

    # Keep the most recent turns within the char budget.
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        extra = len(line) + (2 if kept else 0)
        if total + extra > MAX_CONVERSATION_CHARS and kept:
            break
        kept.insert(0, line)
        total += extra

    truncated = "\n\n".join(kept)
    if len(kept) < len(lines):
        truncated = (
            "[Earlier conversation truncated for length.]\n\n" + truncated
        )
    return truncated, derived_topic
