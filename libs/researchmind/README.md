# researchmind

Local ingest, MemRAG persistence, and retrieval for the ResearchMind agent.

- Scrape web (httpx + trafilatura), PDF (pypdf), DOCX (python-docx)
- Chunk, embed (sentence-transformers), store in SQLite
- Top-k retrieval with graph neighbor expansion and citation formatting

Set `RESEARCHMIND_DATA_DIR` (default `outputs/researchmind`) for the memory database and raw snapshots.
