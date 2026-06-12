from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from researchmind.extract import ExtractedDocument


def extract_pdf(path: Path, *, max_pages: int = 200) -> ExtractedDocument:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages[:max_pages]):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(page_text)

    text = "\n\n".join(pages)
    title = path.stem
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title)

    return ExtractedDocument(
        source_type="pdf",
        uri=str(path.resolve()),
        title=title,
        text=text or path.name,
        mime="application/pdf",
        metadata={"page_count": str(min(len(reader.pages), max_pages))},
    )
