---
name: research-mind
description: Local research agent — scrape, index, and answer with citations
task: research
tools:
  - suggest_urls
  - scrape_web
  - scrape_pdf
  - extract_and_index
  - research_answer
flags:
  auto_search: false
---

## Workflow

### Ingest

1. **Topic only (default):** run `search_urls` (Google + verification) → user confirms URLs → scrape → `extract_and_index`.
2. **Auto search:** when `auto_search` is true, same search pipeline ingests top verified URLs without confirmation.
3. **Direct URL / file:** scrape and index immediately.

### Q&A (offline after ingest)

1. Call `research_answer` with the user question and `session_id`.
2. Retrieve top-k chunks from MemRAG, expand neighbors.
3. Answer using the local model with inline `[n]` citations.
4. Append references from `references/citation-format.md`.

See `references/ingest-modes.md` for mode details.
