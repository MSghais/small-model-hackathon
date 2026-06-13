from echocoach.analysis.fillers import analyze_fillers, highlight_fillers_html


def test_detects_common_fillers():
    text = "So um I think like you know we should basically start."
    analysis = analyze_fillers(text)
    assert analysis.total >= 4
    assert "um" in analysis.counts
    assert "like" in analysis.counts


def test_highlight_wraps_fillers():
    text = "Um hello there."
    analysis = analyze_fillers(text)
    html = highlight_fillers_html(text, analysis)
    assert "<mark" in html
    assert "Um" in html or "um" in html.lower()
