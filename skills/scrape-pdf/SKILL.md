---
name: scrape-pdf
description: Extract text from a local PDF file for indexing
task: research
tools:
  - scrape_pdf
---

## Workflow

1. Receive a path to a `.pdf` file (upload or local path).
2. Call `scrape_pdf` to extract text with pypdf.
3. Pass the `ExtractedDocument` to `extract_and_index`.

See `references/pdf-limits.md` for page limits and scanned-PDF notes.
