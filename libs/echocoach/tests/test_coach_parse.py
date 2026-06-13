from echocoach.coach import parse_coach_response


def test_parse_coach_json_from_fenced_block():
    raw = """Here is feedback:
```json
{
  "summary": "Good energy.",
  "filler_feedback": "Reduce um.",
  "pace_feedback": "Slow down.",
  "rewrite": "We should start now.",
  "one_tip": "Pause after each point."
}
```
"""
    feedback = parse_coach_response(raw)
    assert feedback.summary == "Good energy."
    assert feedback.one_tip == "Pause after each point."
