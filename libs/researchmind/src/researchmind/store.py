from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from researchmind.config import ResearchMindConfig, get_config
from researchmind.embeddings import bytes_to_embedding, embedding_to_bytes


@dataclass(frozen=True)
class StoredDocument:
    id: str
    source_type: str
    uri: str
    title: str
    ingested_at: str
    content_hash: str


@dataclass(frozen=True)
class StoredChunk:
    id: str
    doc_id: str
    ordinal: int
    text: str
    doc_title: str
    doc_uri: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SessionInfo:
    id: str
    topic: str
    created_at: str


class MemRAGStore:
    def __init__(self, config: ResearchMindConfig | None = None) -> None:
        self._config = config or get_config()
        self._config.data_dir.mkdir(parents=True, exist_ok=True)
        (self._config.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        self._db_path = self._config.data_dir / "memory.db"
        self._embed_dim: int | None = None
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            row = self._conn().execute(
                "SELECT dim FROM embed_meta LIMIT 1"
            ).fetchone()
            self._embed_dim = int(row[0]) if row else 384
        return self._embed_dim

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS embed_meta (
                    dim INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    title TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                );
                CREATE TABLE IF NOT EXISTS edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    rel TEXT NOT NULL,
                    PRIMARY KEY (src_id, dst_id, rel)
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
                """
            )

    def set_embed_dim(self, dim: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM embed_meta")
            conn.execute("INSERT INTO embed_meta (dim) VALUES (?)", (dim,))
        self._embed_dim = dim

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def create_session(self, topic: str = "") -> SessionInfo:
        session_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, topic, created_at) VALUES (?, ?, ?)",
                (session_id, topic, created_at),
            )
        return SessionInfo(id=session_id, topic=topic, created_at=created_at)

    def list_sessions(self) -> list[SessionInfo]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, topic, created_at FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [SessionInfo(id=r["id"], topic=r["topic"], created_at=r["created_at"]) for r in rows]

    def get_session(self, session_id: str) -> SessionInfo | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, topic, created_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return SessionInfo(id=row["id"], topic=row["topic"], created_at=row["created_at"])

    def document_exists(self, content_hash: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return row["id"] if row else None

    def find_document_id_by_uri(self, uri: str) -> str | None:
        from researchmind.url_validate import normalize_url

        candidates = [uri.strip()]
        normalized = normalize_url(uri)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
        with self._conn() as conn:
            for candidate in candidates:
                row = conn.execute(
                    "SELECT id FROM documents WHERE uri = ?",
                    (candidate,),
                ).fetchone()
                if row:
                    return str(row["id"])
        return None

    def add_document(
        self,
        *,
        source_type: str,
        uri: str,
        title: str,
        text: str,
        chunks: list[tuple[str, int, str, np.ndarray, dict[str, Any]]],
        session_id: str | None = None,
        raw_snapshot: str | None = None,
    ) -> tuple[str, bool]:
        """Returns (doc_id, was_new). Skips if content_hash already indexed."""
        c_hash = self.content_hash(text)
        existing = self.document_exists(c_hash)
        if existing:
            return existing, False

        doc_id = uuid.uuid4().hex[:12]
        ingested_at = datetime.now(UTC).isoformat()
        if chunks:
            dim = int(chunks[0][3].shape[0])
            self.set_embed_dim(dim)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, source_type, uri, title, ingested_at, content_hash, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, source_type, uri, title, ingested_at, c_hash, session_id),
            )
            for chunk_id, ordinal, chunk_text, emb, meta in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (id, doc_id, ordinal, text, embedding_blob, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        doc_id,
                        ordinal,
                        chunk_text,
                        embedding_to_bytes(emb),
                        json.dumps(meta),
                    ),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO edges (src_id, dst_id, rel) VALUES (?, ?, ?)",
                    (doc_id, chunk_id, "doc_has_chunk"),
                )
            for i in range(len(chunks) - 1):
                conn.execute(
                    "INSERT OR IGNORE INTO edges (src_id, dst_id, rel) VALUES (?, ?, ?)",
                    (chunks[i][0], chunks[i + 1][0], "chunk_next"),
                )

        if raw_snapshot is not None:
            raw_dir = self._config.data_dir / "raw" / doc_id
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "snapshot.txt").write_text(raw_snapshot, encoding="utf-8")

        return doc_id, True

    def list_documents(self, session_id: str | None = None) -> list[StoredDocument]:
        query = "SELECT id, source_type, uri, title, ingested_at, content_hash FROM documents"
        params: tuple[Any, ...] = ()
        if session_id:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY ingested_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            StoredDocument(
                id=r["id"],
                source_type=r["source_type"],
                uri=r["uri"],
                title=r["title"],
                ingested_at=r["ingested_at"],
                content_hash=r["content_hash"],
            )
            for r in rows
        ]

    def get_chunks_with_embeddings(
        self,
        *,
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[StoredChunk, np.ndarray]]:
        dim = self.embed_dim
        query = """
                SELECT c.id, c.doc_id, c.ordinal, c.text, c.embedding_blob, c.meta_json,
                       d.title AS doc_title, d.uri AS doc_uri
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE 1=1
                """
        params: list[Any] = []
        if session_id:
            query += " AND d.session_id = ?"
            params.append(session_id)
        if doc_ids:
            placeholders = ",".join("?" * len(doc_ids))
            query += f" AND d.id IN ({placeholders})"
            params.extend(doc_ids)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        result: list[tuple[StoredChunk, np.ndarray]] = []
        for r in rows:
            chunk = StoredChunk(
                id=r["id"],
                doc_id=r["doc_id"],
                ordinal=r["ordinal"],
                text=r["text"],
                doc_title=r["doc_title"],
                doc_uri=r["doc_uri"],
                metadata=json.loads(r["meta_json"] or "{}"),
            )
            emb = bytes_to_embedding(r["embedding_blob"], dim)
            result.append((chunk, emb))
        return result

    def get_neighbor_chunk_ids(self, chunk_id: str) -> list[str]:
        ids: list[str] = []
        with self._conn() as conn:
            for row in conn.execute(
                "SELECT dst_id FROM edges WHERE src_id = ? AND rel = 'chunk_next'",
                (chunk_id,),
            ):
                ids.append(row["dst_id"])
            for row in conn.execute(
                "SELECT src_id FROM edges WHERE dst_id = ? AND rel = 'chunk_next'",
                (chunk_id,),
            ):
                ids.append(row["src_id"])
        return ids

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[StoredChunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.doc_id, c.ordinal, c.text, c.meta_json,
                       d.title AS doc_title, d.uri AS doc_uri
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        by_id = {
            r["id"]: StoredChunk(
                id=r["id"],
                doc_id=r["doc_id"],
                ordinal=r["ordinal"],
                text=r["text"],
                doc_title=r["doc_title"],
                doc_uri=r["doc_uri"],
                metadata=json.loads(r["meta_json"] or "{}"),
            )
            for r in rows
        }
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        chunk_ids: list[str] | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO session_messages (session_id, role, content, chunk_ids_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(chunk_ids or []),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, chunk_ids_json, created_at
                FROM session_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "chunk_ids": json.loads(r["chunk_ids_json"] or "[]"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def count_chunks(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])
