# Ingest modes

| Mode | `auto_search` | Behavior |
|------|---------------|----------|
| Suggest URLs (confirm) | `false` | Google search + URL verification; user checks boxes before ingest |
| Auto search & ingest | `true` | Same search pipeline; ingests verified URLs without confirmation |
| Direct URL / file | n/a | Skip discovery; ingest provided sources |

Global default: `RESEARCHMIND_AUTO_SEARCH=false`. Gradio dropdown and skill `flags.auto_search` override per run.
