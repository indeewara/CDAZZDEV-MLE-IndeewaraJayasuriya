"""Pydantic schemas for Task 3: the final research report (and, in 3B, the agent hand-off)."""
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Risk(BaseModel):
    title: str = Field(min_length=3)
    evidence: str = Field(min_length=20)   # must cite concrete data, so a bare assertion fails
    severity: Literal["low", "medium", "high"]

    @field_validator("severity", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class HedgeStrategy(BaseModel):
    strategy: str = Field(min_length=3)
    instrument: str = Field(min_length=3)
    sizing: str = Field(min_length=3)      # strikes, notional, stop level - the "data-driven" part
    rationale: str = Field(min_length=20)


class ResearchReport(BaseModel):
    """The three sections the assessment requires, plus provenance."""
    ticker: str
    financial_health_summary: str = Field(min_length=50)
    top_risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy: HedgeStrategy
    data_sources: list[str] = Field(min_length=1)

    def to_markdown(self) -> str:
        lines = [f"# Research report: {self.ticker}", "", "## Financial Health Summary", self.financial_health_summary,
                 "", "## Top Three Risks (next 90 days)"]
        for i, r in enumerate(self.top_risks, 1):
            lines += [f"{i}. **{r.title}** ({r.severity})", f"   Evidence: {r.evidence}"]
        h = self.hedge_strategy
        lines += ["", "## Hedge Strategy Recommendation", f"**{h.strategy}** - {h.instrument}", f"Sizing: {h.sizing}",
                  f"Rationale: {h.rationale}", "", f"*Sources: {', '.join(self.data_sources)}*"]
        return "\n".join(lines)



# ---- 3B: structured hand-offs between the two agents ---------------------------------------------
class DataBrief(BaseModel):
    """Agent A (Data Analyst) -> Agent B (Research Writer). Quantitative only: A has no news access."""
    ticker: str
    current_price: float
    high_52w: float
    low_52w: float
    sma_50: float
    sma_200: float
    rsi_14: float
    pe_ratio: float | None = None
    ytd_return_pct: float | None = None
    annualised_volatility_pct: float
    one_year_avg_volatility_pct: float                # context: is today's volatility high or low for this stock?
    volatility_percentile_1y: float
    expected_90d_move_pct: float
    price_90d_minus_1sigma: float
    price_90d_plus_1sigma: float
    max_drawdown_1y_pct: float
    news_sentiment: str | None = None                 # filled only if A was given headlines to score
    key_findings: list[str] = Field(min_length=3, max_length=5)  # interpretations, each citing its numbers

    @field_validator("key_findings")
    @classmethod
    def _findings_interpret(cls, v):
        # a finding must say something ("price is above both moving averages: uptrend"), not just echo a field
        thin = [f for f in v if len(f.split()) < 6]
        if thin:
            raise ValueError(f"each key finding must be an interpretation of at least 6 words, got: {thin}")
        return v


class ClarificationRequest(BaseModel):
    """Agent B -> Agent A: one specific request for data B cannot get with its own tools."""
    to_agent: Literal["data_analyst"] = "data_analyst"
    question: str = Field(min_length=15)
    reason: str = Field(min_length=15)                # why the report needs it
    headlines: list[str] = Field(default_factory=list, max_length=15)  # material for A's sentiment tool


class ClarificationResponse(BaseModel):
    """Agent A -> Agent B: the requested data, with the tools that produced it."""
    answer: str = Field(min_length=20)
    key_figures: dict[str, float | str] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)


class MultiAgentReport(ResearchReport):
    """Agent B's final report: the three required sections plus how A's clarification was used."""
    clarification_incorporated: str = Field(min_length=20)

    def to_markdown(self) -> str:
        return (super().to_markdown() + "\n\n## How the Data Analyst's clarification was used\n"
                + self.clarification_incorporated)

# ---- Grounding: every number in the report must come from the tool observations ---------------
_NUMBER = re.compile(r"(?<![A-Za-z\d.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")   # 191,753 / 333.63 / 2026; not "Q3"
SMALL_INT = 10          # counts and list numbering ("3 risks", "1-sigma") are not data claims...
REL_TOLERANCE = 0.006   # a quoted figure may be rounded (333.6 for 333.63, 38 for 38.18)
ABS_TOLERANCE = 0.051


def _numbers(text: str) -> list[tuple[str, float, bool]]:
    """(token, value, is_measure): is_measure = written as a percentage or money amount."""
    out = []
    for m in _NUMBER.finditer(text):
        tok = m.group(0)
        before, after = text[max(0, m.start() - 1):m.start()], text[m.end():m.end() + 1]
        out.append((tok, float(tok.replace(",", "")), after == "%" or before in "$€£¥"))
    return out


def ungrounded_numbers(report_text: str, source_text: str) -> list[str]:
    """Numbers in the report that match no number in the sources, allowing for rounding. Empty = grounded.
    Small whole numbers are skipped as counts - unless written as a percentage or money ("4%", "$6")."""
    sources = [v for _, v, _ in _numbers(source_text)]
    bad = []
    for tok, v, is_measure in _numbers(report_text):
        if v <= SMALL_INT and v == int(v) and not is_measure:
            continue
        if any(abs(v - s) <= max(ABS_TOLERANCE, REL_TOLERANCE * s) for s in sources):
            continue
        bad.append(tok)
    return sorted(set(bad), key=lambda t: float(t.replace(",", "")))



def incorporates_clarification(report: "MultiAgentReport", response: ClarificationResponse) -> bool:
    """True if the report's analysis (summary, risks, hedge - NOT the self-describing clarification section) uses at
    least one figure from the Data Analyst's answer. A text-only answer has no figure to trace, so it passes."""
    figures = {tok for tok, v, _ in _numbers(response.answer + " " + json.dumps(response.key_figures))
               if not (v <= SMALL_INT and v == int(v))}
    if not figures:
        return True
    body = report.model_dump_json(exclude={"clarification_incorporated", "data_sources"})
    found = {tok for tok, _, _ in _numbers(body)}
    return bool(figures & found)
