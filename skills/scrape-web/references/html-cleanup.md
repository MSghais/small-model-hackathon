# HTML cleanup

ResearchMind uses **trafilatura** to strip boilerplate and keep main article text.

- `include_tables=true` for data-heavy pages
- `include_comments=false`
- Fallback: first 50k chars of raw HTML if extraction returns empty

Raw snapshot saved under `RESEARCHMIND_DATA_DIR/raw/{doc_id}/snapshot.txt`.
