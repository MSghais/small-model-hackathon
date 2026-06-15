from __future__ import annotations

import json

from agent.models import EducationPptxInput, QuizMakerInput, QuizOutline, QuizQuestion, SlideOutline, SlideSpec


def education_outline_system(skill_body: str) -> str:
    return f"""You are a lesson-planning assistant for teachers.
Follow the skill workflow below and output ONLY valid JSON (no markdown fences).

Skill workflow:
{skill_body}

Required JSON shape (replace every value with real lesson content for the requested topic):
{{
  "title": "Photosynthesis for Grade 6",
  "slides": [
    {{
      "title": "What is photosynthesis?",
      "bullets": ["Plants make food using sunlight", "Happens in chloroplasts"],
      "speaker_note": "Ask students what plants need to grow."
    }}
  ]
}}

Rules:
- Fill in concrete titles and bullets about the user's topic — never copy the example text or type names.
- Use exactly the requested number of content slides (title slide is added separately by the tool).
- At most 3 bullets per slide; each bullet under 12 words.
- speaker_note: one short sentence (under 20 words) or omit.
- Output compact JSON only — no preamble, no markdown fences, stop after the final `}}`.
- When source excerpts are provided, prefer them over general knowledge and keep bullets consistent with those sources.
"""


def outline_max_tokens(slide_count: int) -> int:
    """Cap generation length from slide count so CPU inference does not run to 2048 tokens."""
    count = max(1, min(int(slide_count), 20))
    return min(1024, 100 + count * 130)


def education_outline_user(req: EducationPptxInput, *, source_context: str = "") -> str:
    base = (
        f"Topic: {req.topic}\n"
        f"Grade level: {req.grade}\n"
        f"Number of content slides: {req.slide_count}\n"
    )
    if source_context.strip():
        base += (
            "\nUse the following retrieved source excerpts as factual grounding. "
            "Prefer these over general knowledge when they apply. "
            "Do not invent citations in the JSON output.\n\n"
            f"{source_context}\n"
        )
    if req.conversation_context.strip():
        base += (
            "\nBase the slide outline on this conversation transcript. "
            "Prefer topics and facts discussed over general knowledge.\n\n"
            f"{req.conversation_context.strip()}\n"
        )
    return base + "\nReturn JSON only."


def education_outline_repair(
    invalid_output: str,
    error: str,
    *,
    expected_slides: int | None = None,
) -> str:
    count_line = ""
    if expected_slides is not None:
        count_line = f"\nYou must include exactly {expected_slides} items in the slides array.\n"
    return (
        "The previous response was invalid JSON or did not match the schema.\n"
        f"Validation error: {error}\n"
        f"{count_line}"
        f"Previous output:\n{invalid_output}\n\n"
        "Return corrected JSON only, no explanation."
    )


def outline_to_markdown(title: str, slides: list[dict]) -> str:
    lines = [f"# {title}", ""]
    for index, slide in enumerate(slides, start=1):
        lines.append(f"## Slide {index}: {slide.get('title', 'Untitled')}")
        for bullet in slide.get("bullets", []):
            lines.append(f"- {bullet}")
        note = slide.get("speaker_note", "")
        if note:
            lines.append(f"\n*Teacher note:* {note}")
        lines.append("")
    return "\n".join(lines).strip()


def education_outline_retry_user(req: EducationPptxInput, *, example_json: str) -> str:
    return (
        f"Topic: {req.topic}\n"
        f"Grade level: {req.grade}\n"
        f"Number of content slides: {req.slide_count}\n\n"
        "Your previous response was empty, invalid, or copied schema placeholders. "
        "Write real lesson content for the topic below. "
        "Return ONLY valid JSON matching this structure (replace every value for the topic):\n"
        f"{example_json}"
    )


_SCHEMA_ECHO_MARKERS = (
    "string —",
    "string -",
    "string—",
    "string-",
)


def _looks_like_schema_field(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in _SCHEMA_ECHO_MARKERS):
        return True
    if lowered in {"string", "..."}:
        return True
    return False


def outline_looks_like_schema_echo(outline: SlideOutline) -> bool:
    """True when the model echoed prompt schema placeholders instead of lesson content."""
    if _looks_like_schema_field(outline.title):
        return True

    schema_slides = 0
    for slide in outline.slides:
        if _looks_like_schema_field(slide.title):
            schema_slides += 1
            continue
        bullets = [str(b).strip() for b in slide.bullets if str(b).strip()]
        if bullets and all(_looks_like_schema_field(b) for b in bullets):
            schema_slides += 1
    return schema_slides >= max(1, len(outline.slides) // 2)


def fallback_outline(req: EducationPptxInput) -> SlideOutline:
    """Deterministic outline when the model returns empty or unparseable JSON."""
    topic = req.topic.strip() or "Lesson"
    grade = req.grade
    seeds: list[tuple[str, list[str]]] = [
        ("Introduction", [f"What is {topic}?", f"Overview for grade {grade}"]),
        ("Key concepts", ["Main idea", "Supporting detail"]),
        ("Examples", ["Real-world example", "Classroom activity"]),
        ("Why it matters", ["Connection to students", "Discussion question"]),
        ("Review", ["Summary points", "Check for understanding"]),
        ("Going further", ["Extension idea", "Homework prompt"]),
        ("Vocabulary", ["Important term", "Definition in student language"]),
        ("Wrap-up", ["Recap", "Preview next lesson"]),
    ]
    slides: list[SlideSpec] = []
    for index in range(req.slide_count):
        title, bullets = seeds[index % len(seeds)]
        if index >= len(seeds):
            title = f"{title} ({index + 1})"
        slides.append(
            SlideSpec(
                title=title,
                bullets=bullets,
                speaker_note="Template slide — edit using your lesson sources.",
            )
        )
    return SlideOutline(title=topic[:1].upper() + topic[1:], slides=slides)


def outline_json_example(slide_count: int) -> str:
    example = {
        "title": "Example Lesson",
        "slides": [
            {
                "title": f"Slide {i}",
                "bullets": ["Key point A", "Key point B"],
                "speaker_note": "Brief teaching tip.",
            }
            for i in range(1, slide_count + 1)
        ],
    }
    return json.dumps(example, indent=2)


def quiz_max_tokens(question_count: int) -> int:
    count = max(3, min(int(question_count), 12))
    return min(1536, 120 + count * 180)


def quiz_outline_system(skill_body: str) -> str:
    return f"""You are an expert teacher writing multiple-choice quizzes.
Follow the skill workflow below and output ONLY valid JSON (no markdown fences).

Skill workflow:
{skill_body}

Required JSON shape:
{{
  "title": "Photosynthesis Quiz — Grade 6",
  "instructions": "Read each question. Circle the best answer.",
  "questions": [
    {{
      "prompt": "What do plants use to make food?",
      "choices": ["Sunlight", "Rocks", "Plastic", "Metal"],
      "correct_index": 0,
      "explanation": "Plants use sunlight in photosynthesis."
    }}
  ]
}}

Rules:
- Each question has exactly 4 choices; correct_index is 0-3.
- Grade-appropriate vocabulary and plausible distractors.
- Output compact JSON only — no preamble, no markdown fences.
- When source excerpts are provided, ground questions in those sources.
"""


def quiz_outline_user(req: QuizMakerInput, *, source_context: str = "") -> str:
    base = (
        f"Topic: {req.topic}\n"
        f"Grade level: {req.grade}\n"
        f"Number of questions: {req.question_count}\n"
    )
    if source_context.strip():
        base += (
            "\nUse the following retrieved source excerpts as factual grounding. "
            "Prefer these over general knowledge when they apply.\n\n"
            f"{source_context}\n"
        )
    if req.conversation_context.strip():
        base += (
            "\nBase the quiz on this conversation transcript when relevant.\n\n"
            f"{req.conversation_context.strip()}\n"
        )
    return base + "\nReturn JSON only."


def quiz_outline_repair(
    invalid_output: str,
    error: str,
    *,
    expected_questions: int | None = None,
) -> str:
    count_line = ""
    if expected_questions is not None:
        count_line = f"\nYou must include exactly {expected_questions} items in the questions array.\n"
    return (
        "The previous response was invalid JSON or did not match the QuizOutline schema.\n"
        f"Validation error: {error}\n"
        f"{count_line}"
        f"Previous output:\n{invalid_output}\n\n"
        "Return corrected JSON only, no explanation."
    )


def quiz_outline_retry_user(req: QuizMakerInput, *, example_json: str) -> str:
    return (
        f"Topic: {req.topic}\n"
        f"Grade level: {req.grade}\n"
        f"Number of questions: {req.question_count}\n\n"
        "Your previous response was empty or invalid. "
        "Write real quiz content for the topic. "
        "Return ONLY valid JSON matching this structure:\n"
        f"{example_json}"
    )


def quiz_json_example(question_count: int) -> str:
    example = {
        "title": "Example Quiz",
        "instructions": "Circle the best answer for each question.",
        "questions": [
            {
                "prompt": f"Question {i}?",
                "choices": ["Correct answer", "Distractor A", "Distractor B", "Distractor C"],
                "correct_index": 0,
                "explanation": "Brief teacher note.",
            }
            for i in range(1, question_count + 1)
        ],
    }
    return json.dumps(example, indent=2)


def fallback_quiz(req: QuizMakerInput) -> QuizOutline:
    """Deterministic quiz when the model returns empty or unparseable JSON."""
    topic = req.topic.strip() or "Lesson"
    grade = req.grade
    n = req.question_count
    questions: list[QuizQuestion] = []
    for i in range(1, n + 1):
        questions.append(
            QuizQuestion(
                prompt=f"What is an important idea about {topic} (question {i})?",
                choices=[
                    f"A key fact about {topic}",
                    "An unrelated detail",
                    "A common misconception",
                    "None of these",
                ],
                correct_index=0,
                explanation="Template question — edit using your lesson sources.",
            )
        )
    return QuizOutline(
        title=f"{topic[:1].upper() + topic[1:]} Quiz — Grade {grade}",
        instructions="Read each question carefully. Circle the best answer.",
        questions=questions,
    )


def quiz_to_markdown(outline: QuizOutline) -> str:
    lines = [f"# {outline.title}", ""]
    if outline.instructions.strip():
        lines.extend([outline.instructions.strip(), ""])
    for i, q in enumerate(outline.questions, start=1):
        lines.append(f"## Question {i}")
        lines.append("")
        lines.append(q.prompt)
        lines.append("")
        for label, choice in zip("ABCD", q.choices, strict=True):
            lines.append(f"- **{label}.** {choice}")
        correct = "ABCD"[q.correct_index]
        lines.append("")
        lines.append(f"**Answer:** {correct}")
        if q.explanation.strip():
            lines.append(f"*{q.explanation.strip()}*")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
