try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    _analyzer = None
    VADER_AVAILABLE = False


def analyze_text(text: str | None) -> float | None:
    """Return compound score (-1 to +1) or None if unavailable/empty."""
    if not VADER_AVAILABLE or not text or not text.strip():
        return None
    return _analyzer.polarity_scores(text)["compound"]