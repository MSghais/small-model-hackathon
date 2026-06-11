---
name: education-pptx
description: Create a short lesson PowerPoint from a topic and grade level
task: education
tools:
  - create_pptx
model_hints:
  - minicpm5-1b
  - qwen3b-gguf
---

## Workflow

1. Ask for topic, audience grade, and slide count (3–8 content slides).
2. Produce a JSON outline with `title` and `slides` (each slide has `title`, `bullets`, `speaker_note`).
3. Call `create_pptx` with the validated outline.
4. Return a download link and markdown preview for the teacher.
