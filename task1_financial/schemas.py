"""Pydantic schemas: every LLM response is validated against one of these before use."""
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "negative", "neutral"]
MAX_REASON_CHARS = 400

MIN_SENTENCES, MAX_SENTENCES = 3, 5  # assessment requirement for the signal justification
# Proxy for "reasons over the combination": the justification must touch at least this many
# indicator families. It cannot prove good reasoning, but it rejects single-indicator answers.
MIN_INDICATOR_FAMILIES = 3
INDICATOR_FAMILIES = {
    "moving_averages": ("sma", "moving average", "50-day", "200-day", "golden cross", "death cross"),
    "rsi": ("rsi",),
    "macd": ("macd",),
    "bollinger": ("bollinger", "band", "%b", "percent b"),
}
# sentence end = . ! ? followed by whitespace and a capital/digit, so "3.3%" is not a break
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_BREAK.split(text.strip()) if s.strip()])


class HeadlineSentiment(BaseModel):
    """One headline's sentiment, as returned by the LLM."""
    headline: str = Field(min_length=1)
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _normalise_label(cls, v):
        # tolerate "Positive" / " neutral " - case is noise, not a wrong answer
        return v.strip().lower() if isinstance(v, str) else v


class TradingSignal(BaseModel):
    """LLM Buy/Hold/Sell call reasoned over the technical indicators."""
    signal: Literal["Buy", "Hold", "Sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_conflict: str = Field(min_length=1, max_length=MAX_REASON_CHARS)
    justification: str = Field(min_length=1)

    @field_validator("signal", mode="before")
    @classmethod
    def _normalise_signal(cls, v):
        return v.strip().capitalize() if isinstance(v, str) else v  # "BUY" / "buy" -> "Buy"

    @field_validator("justification")
    @classmethod
    def _check_justification(cls, v: str) -> str:
        n = count_sentences(v)
        if not MIN_SENTENCES <= n <= MAX_SENTENCES:
            raise ValueError(f"justification must be {MIN_SENTENCES}-{MAX_SENTENCES} sentences, got {n}")
        text = v.lower()
        used = [name for name, words in INDICATOR_FAMILIES.items() if any(w in text for w in words)]
        if len(used) < MIN_INDICATOR_FAMILIES:
            raise ValueError(f"justification must weigh at least {MIN_INDICATOR_FAMILIES} indicator "
                             f"families together, only found {used}")
        return v


class SentimentAggregate(BaseModel):
    """Overall news sentiment across all successfully scored headlines."""
    score: float | None = Field(ge=-1.0, le=1.0)  # None when nothing could be scored
    label: Literal["positive", "negative", "neutral", "unknown"]
    counts: dict[str, int]
    n_scored: int
    n_failed: int
