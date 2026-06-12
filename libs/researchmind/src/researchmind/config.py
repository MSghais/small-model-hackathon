from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchMindConfig:
    data_dir: Path
    embed_model: str
    auto_search: bool
    top_k: int
    max_context_chunks: int
    chunk_size: int
    chunk_overlap: int


def get_config() -> ResearchMindConfig:
    data_dir = Path(
        os.environ.get("RESEARCHMIND_DATA_DIR", "outputs/researchmind")
    ).expanduser()
    return ResearchMindConfig(
        data_dir=data_dir,
        embed_model=os.environ.get("RESEARCHMIND_EMBED_MODEL", "all-MiniLM-L6-v2"),
        auto_search=os.environ.get("RESEARCHMIND_AUTO_SEARCH", "false").lower()
        in ("1", "true", "yes"),
        top_k=int(os.environ.get("RESEARCHMIND_TOP_K", "5")),
        max_context_chunks=int(os.environ.get("RESEARCHMIND_MAX_CONTEXT_CHUNKS", "8")),
        chunk_size=int(os.environ.get("RESEARCHMIND_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.environ.get("RESEARCHMIND_CHUNK_OVERLAP", "128")),
    )
