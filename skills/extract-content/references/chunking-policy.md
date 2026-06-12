# Chunking policy

| Setting | Env var | Default |
|---------|---------|---------|
| Chunk size (words) | `RESEARCHMIND_CHUNK_SIZE` | 512 |
| Overlap (words) | `RESEARCHMIND_CHUNK_OVERLAP` | 128 |
| Embedding model | `RESEARCHMIND_EMBED_MODEL` | `all-MiniLM-L6-v2` |

Chunks link via `chunk_next` edges for neighbor expansion at retrieval time.
