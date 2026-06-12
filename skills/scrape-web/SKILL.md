---
name: scrape-web
description: Fetch a web page and extract clean text for indexing
task: research
tools:
  - scrape_web
---

## Workflow

1. Receive a full `https://` URL from the user or orchestrator.
2. Call `scrape_web` with the URL.
3. Return title, extracted text, and final URL metadata.
4. Pass the `ExtractedDocument` to `extract_and_index` for MemRAG storage.

See `references/html-cleanup.md` for extraction settings and `references/allowed-domains.md` for rate-limit notes.
