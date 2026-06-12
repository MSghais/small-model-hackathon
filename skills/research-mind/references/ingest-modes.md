# Ingest modes

| Mode | `auto_search` | Behavior |
|------|---------------|----------|
| Suggest URLs (confirm) | `false` | LLM proposes 3–5 URLs; user checks boxes before ingest |
| Auto search & ingest | `true` | DuckDuckGo top N URLs ingested without confirmation |
| Direct URL / file | n/a | Skip discovery; ingest provided sources |

Global default: `RESEARCHMIND_AUTO_SEARCH=false`. Gradio dropdown and skill `flags.auto_search` override per run.
