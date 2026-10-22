"""
Unit tests for Aspect-Based Sentiment Analysis (ABSA).
Verifies:
- Positive aspect test case
- Negative aspect test case
- Mixed / Contrasting opinions test case ("camera is excellent, but battery life is disappointing")
- No detectable aspect test case
- Clause splitting and multi-aspect returns
"""

import pytest
from src.models.absa_engine import AspectBasedSentimentAnalyzer, analyze_aspect_sentiment


@pytest.fixture
def analyzer():
    return AspectBasedSentimentAnalyzer()


def test_positive_aspect_sentiment(analyzer):
    text = "The screen and display quality is stunning and absolutely amazing."
    res = analyzer.analyze(text)
    assert res["status"] == "success"
    assert res["aspect_count"] >= 1
    aspect_names = [a["aspect"] for a in res["aspects"]]
    assert any(asp in ["screen", "display"] for asp in aspect_names)
    for a in res["aspects"]:
        assert a["sentiment"] == "positive"
        assert a["confidence"] > 0.70


def test_negative_aspect_sentiment(analyzer):
    text = "The battery life is terrible and disappointing."
    res = analyzer.analyze(text)
    assert res["status"] == "success"
    assert res["aspect_count"] >= 1
    aspect_names = [a["aspect"] for a in res["aspects"]]
    assert any("battery" in asp for asp in aspect_names)
    for a in res["aspects"]:
        assert a["sentiment"] == "negative"


def test_mixed_contrasting_aspect_sentiment(analyzer):
    text = "The camera is excellent, but the battery life is disappointing."
    res = analyzer.analyze(text)
    assert res["status"] == "success"
    assert res["aspect_count"] == 2
    assert res["overall_sentiment"] == "mixed"

    aspects_dict = {a["aspect"]: a for a in res["aspects"]}
    assert "camera" in aspects_dict
    assert aspects_dict["camera"]["sentiment"] == "positive"

    assert "battery life" in aspects_dict
    assert aspects_dict["battery life"]["sentiment"] == "negative"
    assert aspects_dict["battery life"]["is_contrasting_clause"] is True


def test_no_detectable_aspects(analyzer):
    text = "Just went for a nice walk in the park on a sunny Sunday afternoon."
    res = analyzer.analyze(text)
    assert res["aspect_count"] == 0
    assert res["overall_sentiment"] == "no_detectable_aspects"
    assert res["aspects"] == []


def test_empty_input(analyzer):
    res = analyzer.analyze("")
    assert res["status"] == "empty_input"
    assert res["aspect_count"] == 0
