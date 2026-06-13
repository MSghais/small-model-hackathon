from __future__ import annotations

from typing import Any


def unwrap_update(value: Any) -> Any:
    """Extract payload from gr.update() return values when calling tab handlers directly."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def update_value(value: Any, default: Any = "") -> Any:
    payload = unwrap_update(value)
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload if payload is not None else default


def update_choices(value: Any) -> list[Any]:
    payload = unwrap_update(value)
    if isinstance(payload, dict):
        return list(payload.get("choices") or [])
    return []


def ok(data: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True}
    if data:
        out.update(data)
    out.update(kwargs)
    return out


def err(message: str, **kwargs: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **kwargs}
