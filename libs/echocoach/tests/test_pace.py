from echocoach.analysis.pace import analyze_pace


def test_ideal_pace_scores_high():
    # 150 words in 60s => 150 WPM
    text = " ".join(["word"] * 150)
    pace = analyze_pace(text, 60.0)
    assert pace.wpm == 150.0
    assert pace.score == 100
    assert pace.label == "Ideal pace"


def test_slow_pace_penalized():
    text = " ".join(["word"] * 50)
    pace = analyze_pace(text, 60.0)
    assert pace.wpm == 50.0
    assert pace.score < 100
    assert "slow" in pace.label.lower()
