"""Tests for quiz-maker skill: JSON parse, fallback, and export smoke."""

from pathlib import Path

from agent.models import QuizMakerInput, QuizOutline, QuizQuestion
from agent.prompts import fallback_quiz, quiz_max_tokens, quiz_to_markdown
from agent.runner import AgentRunner
from agent.tools.quiz import create_quiz, create_quiz_docx, create_quiz_html


def test_quiz_max_tokens_scales_with_question_count():
    assert quiz_max_tokens(5) == 1020
    assert quiz_max_tokens(3) == 660
    assert quiz_max_tokens(12) == 1536


def test_parse_quiz_outline_normalizes_count():
    runner = AgentRunner()
    raw = (
        '{"title": "Science Quiz", "instructions": "Circle one.", "questions": ['
        '{"prompt": "Q1?", "choices": ["a", "b", "c", "d"], "correct_index": 0, "explanation": "e1"},'
        '{"prompt": "Q2?", "choices": ["a", "b", "c", "d"], "correct_index": 1, "explanation": "e2"},'
        '{"prompt": "Q3?", "choices": ["a", "b", "c", "d"], "correct_index": 2, "explanation": "e3"}'
        "]}"
    )
    outline = runner._parse_quiz_outline(raw, expected_questions=5)
    assert len(outline.questions) == 5
    assert outline.title == "Science Quiz"


def test_parse_quiz_outline_trims_extra_questions():
    runner = AgentRunner()
    questions = ",".join(
        f'{{"prompt": "Q{i}?", "choices": ["a","b","c","d"], "correct_index": 0, "explanation": ""}}'
        for i in range(1, 8)
    )
    raw = f'{{"title": "Long", "questions": [{questions}]}}'
    outline = runner._parse_quiz_outline(raw, expected_questions=5)
    assert len(outline.questions) == 5


def test_parse_quiz_outline_or_error_empty():
    runner = AgentRunner()
    outline, err = runner._parse_quiz_outline_or_error("", 5, None)
    assert outline is None
    assert "empty" in err.lower()


def test_fallback_quiz_has_requested_count():
    req = QuizMakerInput(topic="Fractions", grade="5", question_count=7)
    outline = fallback_quiz(req)
    assert len(outline.questions) == 7
    assert "Fractions" in outline.title
    assert all(len(q.choices) == 4 for q in outline.questions)


def test_quiz_to_markdown_includes_answers():
    outline = QuizOutline(
        title="Test",
        instructions="Read carefully.",
        questions=[
            QuizQuestion(
                prompt="2+2?",
                choices=["4", "3", "5", "6"],
                correct_index=0,
                explanation="Basic addition.",
            )
        ],
    )
    md = quiz_to_markdown(outline)
    assert "2+2?" in md
    assert "**Answer:** A" in md


def test_create_quiz_docx_and_html(tmp_path: Path):
    outline = QuizOutline(
        title="Smoke Quiz",
        instructions="Circle the best answer.",
        questions=[
            QuizQuestion(
                prompt="Sample?",
                choices=["Yes", "No", "Maybe", "Sometimes"],
                correct_index=0,
                explanation="Because.",
            ),
            QuizQuestion(
                prompt="Another?",
                choices=["A", "B", "C", "D"],
                correct_index=2,
                explanation="C is correct.",
            ),
            QuizQuestion(
                prompt="Third?",
                choices=["1", "2", "3", "4"],
                correct_index=1,
                explanation="Two.",
            ),
        ],
    )
    docx_path = tmp_path / "quiz.docx"
    html_path = tmp_path / "quiz.html"
    create_quiz_docx(outline, docx_path)
    create_quiz_html(outline, html_path)
    assert docx_path.stat().st_size > 100
    html_text = html_path.read_text(encoding="utf-8")
    assert "Smoke Quiz" in html_text
    assert "Answer key" in html_text

    paths = create_quiz(outline, tmp_path / "out", stem="worksheet")
    assert paths["docx"].exists()
    assert paths["html"].exists()
