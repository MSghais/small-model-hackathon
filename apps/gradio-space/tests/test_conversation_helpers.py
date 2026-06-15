from gradio_space.conversation_helpers import format_conversation_context


def test_format_research_history():
    history = [
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Plants convert light into energy."},
    ]
    text, topic = format_conversation_context(history, "research")
    assert "User: What is photosynthesis?" in text
    assert "Assistant: Plants convert light" in text
    assert topic == "What is photosynthesis?"


def test_format_gradio_history():
    history = [["Hello", "Hi there"], ["Explain mitosis", "Cells divide."]]
    text, topic = format_conversation_context(history, "gradio")
    assert "User: Hello" in text
    assert "Assistant: Hi there" in text
    assert "User: Explain mitosis" in text
    assert topic == "Hello"


def test_format_voice_role_objects():
    history = [
        {"role": "user", "content": "Teach fractions"},
        {"role": "assistant", "content": "Fractions are parts of a whole."},
    ]
    text, topic = format_conversation_context(history, "voice")
    assert "User: Teach fractions" in text
    assert topic == "Teach fractions"


def test_format_debug_pairs():
    history = [["Debug question", "Debug answer"]]
    text, topic = format_conversation_context(history, "debug")
    assert "User: Debug question" in text
    assert topic == "Debug question"


def test_truncates_long_history():
    long_user = "x" * 5000
    history = [
        {"role": "user", "content": long_user},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    text, _topic = format_conversation_context(history, "research")
    assert len(text) <= 8200
    assert "recent question" in text
    assert "truncated" in text.lower() or "recent answer" in text
