# Multiple-choice format

## Question structure

Each question must include:

- `prompt`: clear stem (one sentence or short paragraph)
- `choices`: array of exactly **4** strings (no A/B/C/D prefixes in JSON)
- `correct_index`: integer 0–3 (index into `choices`)
- `explanation`: one or two sentences for teachers (why the answer is correct)

## Distractors

- All four options should be plausible at the target grade level.
- Avoid "all of the above" / "none of the above" unless topic-specific.
- Keep choice length similar within a question.

## Quiz outline

- `title`: quiz title (include topic and grade when helpful)
- `instructions`: student-facing directions (e.g. "Circle the best answer.")
- `questions`: 3–12 items; default request is 5–10
