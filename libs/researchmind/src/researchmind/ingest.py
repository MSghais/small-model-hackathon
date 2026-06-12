from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from researchmind.chunking import chunk_text
from researchmind.config import ResearchMindConfig, get_config
from researchmind.embeddings import embed_texts
from researchmind.extract import ExtractedDocument, extract_docx
from researchmind.scrape_pdf import extract_pdf
from researchmind.scrape_web import fetch_and_extract
from researchmind.store import MemRAGStore


class IngestPipeline:
    def __init__(
        self,
        store: MemRAGStore | None = None,
        config: ResearchMindConfig | None = None,
    ) -> None:
        self._config = config or get_config()
        self._store = store or MemRAGStore(self._config)

    @property
    def store(self) -> MemRAGStore:
        return self._store

    def ingest_document(
        self,
        doc: ExtractedDocument,
        *,
        session_id: str | None = None,
        raw_snapshot: str | None = None,
    ) -> tuple[str, bool]:
        doc_id_prefix = self._store.content_hash(doc.text)[:12]
        chunks = chunk_text(
            doc.text,
            doc_id=doc_id_prefix,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        if not chunks and doc.text.strip():
            from researchmind.chunking import TextChunk

            chunks = [
                TextChunk(
                    chunk_id=f"{doc_id_prefix}_0",
                    ordinal=0,
                    text=doc.text[: self._config.chunk_size],
                )
            ]

        chunks_text = [c.text for c in chunks]
        embeddings = embed_texts(chunks_text, model_name=self._config.embed_model)
        chunk_tuples: list[tuple[str, int, str, np.ndarray, dict[str, Any]]] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk_tuples.append(
                (
                    chunk.chunk_id,
                    chunk.ordinal,
                    chunk.text,
                    emb,
                    {"source_type": doc.source_type},
                )
            )

        return self._store.add_document(
            source_type=doc.source_type,
            uri=doc.uri,
            title=doc.title,
            text=doc.text,
            chunks=chunk_tuples,
            session_id=session_id,
            raw_snapshot=raw_snapshot or doc.text[:100_000],
        )

    def ingest_url(self, url: str, *, session_id: str | None = None) -> tuple[str, bool]:
        doc = fetch_and_extract(url)
        return self.ingest_document(doc, session_id=session_id, raw_snapshot=doc.text)

    def ingest_pdf(self, path: Path, *, session_id: str | None = None) -> tuple[str, bool]:
        doc = extract_pdf(path)
        return self.ingest_document(doc, session_id=session_id)

    def ingest_docx(self, path: Path, *, session_id: str | None = None) -> tuple[str, bool]:
        doc = extract_docx(path)
        return self.ingest_document(doc, session_id=session_id)

    def ingest_path(self, path: Path, *, session_id: str | None = None) -> tuple[str, bool]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.ingest_pdf(path, session_id=session_id)
        if suffix == ".docx":
            return self.ingest_docx(path, session_id=session_id)
        text = path.read_text(encoding="utf-8", errors="replace")
        doc = ExtractedDocument(
            source_type="file",
            uri=str(path.resolve()),
            title=path.stem,
            text=text,
            mime="text/plain",
        )
        return self.ingest_document(doc, session_id=session_id)
