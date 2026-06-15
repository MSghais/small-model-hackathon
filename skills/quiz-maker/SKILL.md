---
name: quiz-maker
description: Create a multiple-choice quiz from a topic and grade level
task: education
tools:
  - create_quiz
model_hints:
  - minicpm5-1b
---

# Quiz maker

Generate a printable multiple-choice quiz (worksheet + answer key) for a topic and grade level.

## Workflow

1. Gather optional source context (web URLs, uploaded files, or session RAG).
2. Produce a `QuizOutline` JSON object with 3–12 questions (typically 5–10).
3. Export DOCX (student worksheet + answer key page) and HTML preview via `create_quiz`.

## Output rules

- Each question has exactly **4** choices labeled A–D.
- Exactly one correct answer per question (`correct_index` 0–3).
- Include a short explanation for each answer (teacher reference).
- Grade-appropriate vocabulary and distractors.
- Ground content in provided sources when available.

See `references/mcq-format.md` for MCQ structure details.
