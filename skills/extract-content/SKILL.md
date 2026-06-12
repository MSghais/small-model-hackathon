---
name: extract-content
description: Chunk, embed, and index extracted text into MemRAG
task: research
tools:
  - extract_and_index
---

## Workflow

1. Receive an `ExtractedDocument` (from web, PDF, or DOCX scrape).
2. Call `extract_and_index` with optional `session_id`.
3. Chunks are embedded with sentence-transformers and stored in SQLite.
4. Duplicate content (same hash) is skipped.

See `references/chunking-policy.md` for chunk size and overlap defaults.
