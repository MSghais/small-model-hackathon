from __future__ import annotations

import html
from typing import Any


def _icon_for_source(source_type: str) -> str:
    st = (source_type or "").lower()
    if st in ("web", "url", "scrape"):
        return "language"
    if st == "pdf":
        return "picture_as_pdf"
    return "description"


def render_doc_cards(documents: list[dict[str, Any]], *, rag_active: bool) -> str:
    if not documents:
        return (
            '<p class="studio-empty-docs">No documents indexed yet. Paste a URL or upload a file.</p>'
        )
    badge = (
        '<span class="studio-badge studio-badge-rag">RAG Active</span>'
        if rag_active
        else '<span class="studio-badge studio-badge-muted">No sources</span>'
    )
    cards: list[str] = []
    for doc in documents:
        title = html.escape(str(doc.get("title") or "Untitled"))
        meta = html.escape(str(doc.get("meta") or ""))
        icon = _icon_for_source(str(doc.get("source_type") or ""))
        cards.append(
            f"""
<div class="studio-doc-card" data-doc-id="{html.escape(str(doc.get("id", "")))}">
  <span class="material-symbols-outlined studio-doc-icon">{icon}</span>
  <div class="studio-doc-body">
    <p class="studio-doc-title">{title}</p>
    <p class="studio-doc-meta">{meta}</p>
  </div>
</div>"""
        )
    return f'<div class="studio-doc-header">{badge}</div><div class="studio-doc-list">{"".join(cards)}</div>'


def render_slide_canvas(preview_html: str, *, empty_message: str | None = None) -> str:
    if not preview_html or not preview_html.strip():
        msg = html.escape(empty_message or "Generate slides to preview your lesson here.")
        return f'<div class="studio-canvas-empty"><p>{msg}</p></div>'
    return f'<div class="studio-canvas-inner">{preview_html}</div>'


def render_echo_coach_panel(
    *,
    pace_score: int | None = None,
    wpm: float | None = None,
    tip: str | None = None,
    report_md: str | None = None,
    listening: bool = False,
) -> str:
    if listening:
        return """
<div class="studio-coach-panel studio-coach-live">
  <div class="studio-coach-header">
    <span class="studio-coach-dot"></span>
    <span class="studio-coach-label">Recording…</span>
  </div>
  <p class="studio-coach-hint">Speak your lesson, then analyze for pace and filler feedback.</p>
</div>"""

    if pace_score is None and not tip and not report_md:
        return """
<div class="studio-coach-panel studio-coach-idle">
  <p class="studio-coach-hint">Record a pitch in the Coach view, then click <strong>Analyze pitch</strong> for metrics.</p>
</div>"""

    score = pace_score if pace_score is not None else "—"
    pace = f"{wpm:.0f}" if wpm is not None else "—"
    tip_html = html.escape(tip or "")
    report_block = ""
    if report_md:
        safe = html.escape(report_md[:600])
        report_block = f'<div class="studio-coach-report">{safe}</div>'

    return f"""
<div class="studio-coach-panel studio-coach-results">
  <div class="studio-coach-header">
    <span class="studio-coach-label">Analysis results</span>
    <span class="studio-coach-tag">EchoCoach</span>
  </div>
  <div class="studio-coach-metrics">
    <div class="studio-metric">
      <p class="studio-metric-label">Pace score</p>
      <p class="studio-metric-value">{score}</p>
    </div>
    <div class="studio-metric">
      <p class="studio-metric-label">Pace (WPM)</p>
      <p class="studio-metric-value">{pace}</p>
    </div>
  </div>
  {f'<p class="studio-coach-tip">{tip_html}</p>' if tip_html else ''}
  {report_block}
</div>"""
