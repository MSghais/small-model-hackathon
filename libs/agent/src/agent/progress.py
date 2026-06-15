from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from time import monotonic
from typing import Any, Callable

ProgressUpdateFn = Callable[[float, str], None]
ProgressStepFn = Callable[["ProgressStep"], None]

# Typical share of wall time per phase (used for ETA while a step is running).
_STEP_WEIGHTS: dict[str, float] = {
    "load_model": 0.12,
    "gather_sources": 0.18,
    "generate_outline": 0.45,
    "repair_outline": 0.12,
    "create_exports": 0.06,
    "render_previews": 0.07,
}


@dataclass
class ProgressStep:
    name: str
    label: str
    started_at: float
    ended_at: float | None = None
    detail: str = ""

    @property
    def duration_s(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at


@dataclass
class SlideGenerationProgress:
    """Tracks slide-generation phases with timing and optional Gradio updates."""

    on_update: ProgressUpdateFn | None = None
    on_step: ProgressStepFn | None = None
    steps: list[ProgressStep] = field(default_factory=list)
    started_at: float = field(default_factory=monotonic)
    _current: ProgressStep | None = field(default=None, repr=False)
    _completed_weight: float = field(default=0.0, repr=False)

    def begin(self, name: str, label: str, *, detail: str = "") -> None:
        self._finish_current()
        step = ProgressStep(
            name=name,
            label=label,
            started_at=monotonic(),
            detail=detail,
        )
        self._current = step
        self.steps.append(step)
        if self.on_step is not None:
            self.on_step(step)
        self._emit(label, detail)

    def detail(self, detail: str) -> None:
        if self._current is not None:
            self._current.detail = detail
            self._emit(self._current.label, detail)

    def finish(self) -> None:
        self._finish_current()
        self._emit("Done", "")

    def elapsed_s(self) -> float:
        return monotonic() - self.started_at

    def estimate_remaining_s(self) -> float | None:
        elapsed = self.elapsed_s()
        if elapsed < 0.5 or not self.steps:
            return None

        done_weight = self._completed_weight
        current = self._current
        if current is not None:
            step_weight = _STEP_WEIGHTS.get(current.name, 0.08)
            step_elapsed = monotonic() - current.started_at
            if step_elapsed > 0.2:
                done_weight += step_weight * min(0.85, step_elapsed / max(step_elapsed + 8.0, 1.0))

        total_weight = sum(_STEP_WEIGHTS.values())
        if done_weight <= 0.05:
            return None
        progress_ratio = min(done_weight / total_weight, 0.95)
        projected_total = elapsed / progress_ratio
        remaining = projected_total - elapsed
        return max(0.0, remaining)

    def format_log(self, *, include_eta: bool = True) -> str:
        lines: list[str] = []
        elapsed = self.elapsed_s()
        lines.append(f"**Elapsed:** {elapsed:.1f}s")

        if include_eta:
            remaining = self.estimate_remaining_s()
            if remaining is not None:
                lines.append(f"**Est. remaining:** ~{remaining:.0f}s")

        lines.append("")
        for index, step in enumerate(self.steps, start=1):
            icon = "✓" if step.ended_at is not None else "…"
            duration = ""
            if step.duration_s is not None:
                duration = f" ({step.duration_s:.1f}s)"
            line = f"{index}. {icon} **{step.label}**{duration}"
            if step.detail:
                line += f" — {step.detail}"
            lines.append(line)

        return "\n".join(lines)

    def format_log_html(
        self,
        *,
        running: bool = False,
        footer_html: str = "",
    ) -> str:
        elapsed = self.elapsed_s()
        eta = self.estimate_remaining_s() if running else None
        banner = (
            '<div class="slide-gen-log-banner running">Generating slides…</div>'
            if running
            else '<div class="slide-gen-log-banner done">Generation complete</div>'
        )
        eta_html = (
            f'<div class="slide-gen-log-meta">Est. remaining: ~{int(eta)}s</div>'
            if eta is not None and running
            else ""
        )
        steps_html: list[str] = []
        for step in self.steps:
            done = step.ended_at is not None
            status = "done" if done else "active"
            icon = "✓" if done else "●"
            duration = (
                f' <span class="slide-gen-log-dur">({step.duration_s:.1f}s)</span>'
                if step.duration_s is not None
                else ""
            )
            detail = (
                f' <span class="slide-gen-log-detail">— {escape(step.detail)}</span>'
                if step.detail
                else ""
            )
            steps_html.append(
                f'<li class="slide-gen-log-step {status}">'
                f'<span class="slide-gen-log-icon">{icon}</span>'
                f'<span class="slide-gen-log-label">{escape(step.label)}</span>'
                f"{duration}{detail}</li>"
            )
        steps_block = (
            f'<ol class="slide-gen-log-steps">{"".join(steps_html)}</ol>'
            if steps_html
            else '<p class="slide-gen-log-empty">Waiting for first step…</p>'
        )
        return (
            f'<div class="slide-gen-log">'
            f"{banner}"
            f'<div class="slide-gen-log-meta">Elapsed: {elapsed:.1f}s</div>'
            f"{eta_html}"
            f"{steps_block}"
            f"{footer_html}"
            f"</div>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_s": round(self.elapsed_s(), 2),
            "estimate_remaining_s": (
                round(remaining, 1)
                if (remaining := self.estimate_remaining_s()) is not None
                else None
            ),
            "steps": [
                {
                    "name": step.name,
                    "label": step.label,
                    "detail": step.detail,
                    "duration_s": (
                        round(step.duration_s, 2) if step.duration_s is not None else None
                    ),
                    "status": "done" if step.ended_at is not None else "running",
                }
                for step in self.steps
            ],
        }

    def _finish_current(self) -> None:
        if self._current is None or self._current.ended_at is not None:
            return
        self._current.ended_at = monotonic()
        self._completed_weight += _STEP_WEIGHTS.get(self._current.name, 0.08)

    def _emit(self, label: str, detail: str) -> None:
        if self.on_update is None:
            return
        total_weight = sum(_STEP_WEIGHTS.values())
        fraction = min(self._completed_weight / total_weight, 0.98)
        desc = label if not detail else f"{label} — {detail}"
        self.on_update(fraction, desc)


@dataclass
class QuizGenerationProgress(SlideGenerationProgress):
    """Quiz generation progress tracker (same steps, quiz-specific banner text)."""

    def format_log_html(
        self,
        *,
        running: bool = False,
        footer_html: str = "",
    ) -> str:
        elapsed = self.elapsed_s()
        eta = self.estimate_remaining_s() if running else None
        banner = (
            '<div class="slide-gen-log-banner running">Generating quiz…</div>'
            if running
            else '<div class="slide-gen-log-banner done">Quiz generation complete</div>'
        )
        eta_html = (
            f'<div class="slide-gen-log-meta">Est. remaining: ~{int(eta)}s</div>'
            if eta is not None and running
            else ""
        )
        steps_html: list[str] = []
        for step in self.steps:
            done = step.ended_at is not None
            status = "done" if done else "active"
            icon = "✓" if done else "●"
            duration = (
                f' <span class="slide-gen-log-dur">({step.duration_s:.1f}s)</span>'
                if step.duration_s is not None
                else ""
            )
            detail = (
                f' <span class="slide-gen-log-detail">— {escape(step.detail)}</span>'
                if step.detail
                else ""
            )
            steps_html.append(
                f'<li class="slide-gen-log-step {status}">'
                f'<span class="slide-gen-log-icon">{icon}</span>'
                f'<span class="slide-gen-log-label">{escape(step.label)}</span>'
                f"{duration}{detail}</li>"
            )
        steps_block = (
            f'<ol class="slide-gen-log-steps">{"".join(steps_html)}</ol>'
            if steps_html
            else '<p class="slide-gen-log-empty">Waiting for first step…</p>'
        )
        return (
            f'<div class="slide-gen-log">'
            f"{banner}"
            f'<div class="slide-gen-log-meta">Elapsed: {elapsed:.1f}s</div>'
            f"{eta_html}"
            f"{steps_block}"
            f"{footer_html}"
            f"</div>"
        )
