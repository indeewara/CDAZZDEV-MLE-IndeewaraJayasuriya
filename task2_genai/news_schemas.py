"""Task 2 output contract: the Pydantic schema every analysis must satisfy, plus the grounding
check (are all names and numbers in the input?). Used to filter generated data and again in
evaluation as an automatic hallucination signal."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EVENT_TYPES = ("earnings", "guidance", "analyst_rating", "m_and_a", "capital_return", "insider_transaction",
               "regulatory", "litigation", "product", "management_change", "macro")
DIRECTIONS = ("positive", "negative", "neutral")
MAGNITUDES = ("low", "medium", "high")

MIN_RATIONALE_SENTENCES, MAX_RATIONALE_SENTENCES = 2, 3
MAX_RATIONALE_CHARS = 600
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")  # "5.2%" is not a break
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_BREAK.split(text.strip()) if s.strip()])


def _normalise_label(v):
    """'M&A' -> 'm_and_a', ' Negative ' -> 'negative'. Case and spacing are noise, not a wrong answer."""
    return v.strip().lower().replace(" ", "_").replace("&", "_and_") if isinstance(v, str) else v


class LabelFields(BaseModel):
    """Just the three label fields. Used to read the blind labeller's answer, which exists only to
    cross-check the teacher's labels - its rationale and entities are never used."""
    event_type: Literal[EVENT_TYPES]
    impact_direction: Literal[DIRECTIONS]
    magnitude: Literal[MAGNITUDES]

    _norm = field_validator("event_type", "impact_direction", "magnitude", mode="before")(
        lambda cls, v: _normalise_label(v))


class NewsAnalysis(BaseModel):
    """Exactly the six keys from the problem statement; anything else is a validation error."""
    model_config = ConfigDict(extra="forbid")

    event_type: Literal[EVENT_TYPES]
    primary_entity: str = Field(min_length=1)
    mentioned_entities: list[str]
    impact_direction: Literal[DIRECTIONS]
    magnitude: Literal[MAGNITUDES]
    rationale: str = Field(min_length=1, max_length=MAX_RATIONALE_CHARS)

    _norm = field_validator("event_type", "impact_direction", "magnitude", mode="before")(
        lambda cls, v: _normalise_label(v))

    @field_validator("rationale")
    @classmethod
    def _check_rationale(cls, v: str) -> str:
        n = count_sentences(v)
        if not MIN_RATIONALE_SENTENCES <= n <= MAX_RATIONALE_SENTENCES:
            raise ValueError(f"rationale must be {MIN_RATIONALE_SENTENCES}-{MAX_RATIONALE_SENTENCES} sentences, got {n}")
        return v


def grounding_issues(news_text: str, a: NewsAnalysis) -> list[str]:
    """Names and numbers in the analysis that do not appear in the input text.
    Empty list = grounded. This is a strict, automatic proxy for hallucination."""
    text = news_text.lower()
    issues = [f"entity not in input: {e!r}" for e in a.mentioned_entities if e.strip().lower() not in text]
    if a.event_type != "macro" and a.primary_entity.strip().lower() not in text:
        issues.append(f"primary_entity not in input: {a.primary_entity!r}")
    numbers_in_text = set(_NUMBER.findall(news_text))
    issues += [f"number not in input: {n}" for n in _NUMBER.findall(a.rationale) if n not in numbers_in_text]
    return issues


VERDICTS = ("correct", "partially_correct", "incorrect", "hallucinated")


class JudgeScore(BaseModel):
    """Structured output of the LLM-as-judge (Task 2C)."""
    label_accuracy: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    rationale_quality: int = Field(ge=1, le=5)
    format_compliance: int = Field(ge=1, le=5)
    verdict: Literal[VERDICTS]
    justification: str = Field(min_length=1)

    _norm = field_validator("verdict", mode="before")(lambda cls, v: _normalise_label(v))
