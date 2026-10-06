"""Task 1B - LLM sentiment and signal reasoning on top of the Task 1A pipeline (Groq API)."""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import APIError, Groq
from pydantic import BaseModel, ValidationError

from data_pipeline import TRADING_DAYS_PER_YEAR, _num
from prompts import (HEADLINE_SENTIMENT_SYSTEM, HEADLINE_SENTIMENT_USER, REPAIR_USER,
                     TRADING_SIGNAL_NEWS_ADDENDUM, TRADING_SIGNAL_SYSTEM, TRADING_SIGNAL_USER)
from schemas import (MAX_SENTENCES, MIN_INDICATOR_FAMILIES, MIN_SENTENCES, HeadlineSentiment,
                     SentimentAggregate, TradingSignal)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
VALIDATION_LOG = OUTPUT_DIR / "llm_validation.log"

# openai/gpt-oss-120b: strongest text model on Groq's free tier, supports JSON mode.
# Override with GROQ_MODEL in .env if Groq retires it (it already retired llama-3.3-70b).
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0          # classification: we want the same label on every run
# gpt-oss reasons before answering. "low" is enough to label one headline; the signal has to
# weigh conflicting indicators, so it gets "medium".
SENTIMENT_REASONING_EFFORT = "low"
SIGNAL_REASONING_EFFORT = "medium"
MAX_ATTEMPTS = 3         # 1 call + 2 repair retries, then give up on that headline

# Technical context for the signal prompt
TREND_LOOKBACK = 10        # trading days (~2 weeks) used for slopes and "10 days ago" comparisons
SHORT_RETURN_DAYS, MEDIUM_RETURN_DAYS = 20, 60   # ~1 and ~3 months
SQUEEZE_PERCENTILE = 20    # band width in the bottom 20% of its 1y range = Bollinger squeeze

POLARITY = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
# |score| below this is reported as neutral: a few weakly positive headlines are not a signal
SENTIMENT_LABEL_BAND = 0.15


def get_client() -> Groq:
    """Groq client; the key comes from the environment or the repo-root .env, never from code."""
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set - add it to .env (see .env.example)")
    return Groq()


def call_llm_json(client: Groq, system: str, user: str, schema: type[BaseModel],
                  reasoning_effort: str = SENTIMENT_REASONING_EFFORT) -> BaseModel | None:
    """Ask for JSON, validate against `schema`, and on failure send the validation errors back
    for a corrected reply. Every failure is logged; returns None if all attempts fail, so one
    bad response never crashes the pipeline."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=TEMPERATURE,
                reasoning_effort=reasoning_effort, response_format={"type": "json_object"},
            )
        except APIError as exc:  # rate limit, server error, JSON-mode rejection, ...
            logger.warning("LLM API error (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc)
            continue

        try:
            raw = resp.choices[0].message.content or ""
        except (IndexError, AttributeError):  # no choices / malformed envelope -> treat as invalid reply
            raw = ""
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("Validation failed (attempt %d/%d) for %s: %s | raw=%r",
                           attempt, MAX_ATTEMPTS, schema.__name__, exc.errors(include_url=False), raw[:500])
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": REPAIR_USER.format(errors=exc)}]

    logger.error("Giving up after %d attempts: %s", MAX_ATTEMPTS, user[:200])
    return None


def score_headlines(headlines: list[dict], ticker: str, company: str | None,
                    client: Groq | None = None) -> tuple[list[HeadlineSentiment], list[str]]:
    """One LLM call per headline. Returns (validated results, headlines that failed)."""
    client = client or get_client()
    scored, failed = [], []
    for h in headlines:
        title = h["title"]
        user = HEADLINE_SENTIMENT_USER.format(company=company or ticker, ticker=ticker, headline=title)
        result = call_llm_json(client, HEADLINE_SENTIMENT_SYSTEM, user, HeadlineSentiment)
        if result is None:
            failed.append(title)
            continue
        if result.headline != title:
            # the model paraphrased; keep the true source headline so results stay traceable
            logger.info("LLM altered headline text; restoring original: %r", title)
            result = result.model_copy(update={"headline": title})
        scored.append(result)
    return scored, failed


def aggregate_sentiment(scored: list[HeadlineSentiment], n_failed: int = 0) -> SentimentAggregate:
    """Confidence-weighted mean polarity in [-1, 1]:  sum(polarity_i * conf_i) / sum(conf_i).

    - Confidence is the weight, so an unsure call moves the score less than a sure one.
    - Neutral headlines add weight but zero polarity, so they pull the score toward 0
      (mostly-routine news should not read as a strong signal).
    - Failed headlines are excluded, not counted as neutral, and reported in n_failed.
    """
    counts = {label: sum(s.sentiment == label for s in scored) for label in POLARITY}
    total_weight = sum(s.confidence for s in scored)
    if not scored or total_weight == 0:
        return SentimentAggregate(score=None, label="unknown", counts=counts, n_scored=len(scored), n_failed=n_failed)

    score = sum(POLARITY[s.sentiment] * s.confidence for s in scored) / total_weight
    label = ("positive" if score > SENTIMENT_LABEL_BAND
             else "negative" if score < -SENTIMENT_LABEL_BAND else "neutral")
    return SentimentAggregate(score=round(score, 4), label=label, counts=counts,
                              n_scored=len(scored), n_failed=n_failed)


def _pct(a, b):
    """Percent difference of a over b, None if either side is missing."""
    return _num((a / b - 1) * 100) if a is not None and b else None


def _last_cross(diff: pd.Series) -> dict | None:
    """Most recent sign change of `diff` (e.g. SMA50 - SMA200): direction and trading days ago."""
    sign = np.sign(diff.dropna())
    sign = sign[sign != 0]
    flips = sign.ne(sign.shift()) & sign.shift().notna()
    if not flips.any():
        return None
    when = flips[flips].index[-1]
    return {"direction": "bullish" if sign.loc[when] > 0 else "bearish",
            "days_ago": int(len(sign.loc[when:]) - 1)}


def build_technical_context(df: pd.DataFrame) -> dict | None:
    """Raw indicator levels plus the relationships between them (distance, slope, recency,
    band position). Giving the model relationships, not just levels, is what lets it reason
    over the combination instead of reading values back. None if there is no data."""
    if df.empty or "SMA_200" not in df:
        return None
    last = df.iloc[-1]
    back = df.iloc[-1 - TREND_LOOKBACK] if len(df) > TREND_LOOKBACK else pd.Series(dtype=float)
    close = _num(last["Close"])
    sma50, sma200 = _num(last["SMA_50"]), _num(last["SMA_200"])
    upper, lower, mid = _num(last["BB_upper"]), _num(last["BB_lower"]), _num(last["BB_mid"])

    width = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"] * 100
    recent_width = width.tail(TRADING_DAYS_PER_YEAR).dropna()
    width_pctile = _num((recent_width <= recent_width.iloc[-1]).mean() * 100) if not recent_width.empty else None

    def ret(days):
        return _pct(close, _num(df["Close"].iloc[-1 - days])) if len(df) > days else None

    return {
        "price": {"close": close, "return_1m_pct": ret(SHORT_RETURN_DAYS), "return_3m_pct": ret(MEDIUM_RETURN_DAYS)},
        "trend": {
            "sma50": sma50, "sma200": sma200,
            "price_vs_sma50_pct": _pct(close, sma50),
            "price_vs_sma200_pct": _pct(close, sma200),
            "sma50_vs_sma200_pct": _pct(sma50, sma200),
            f"sma50_slope_{TREND_LOOKBACK}d_pct": _pct(sma50, _num(back.get("SMA_50"))),
            "last_sma50_sma200_cross": _last_cross(df["SMA_50"] - df["SMA_200"]),
        },
        "momentum": {
            "rsi14": _num(last["RSI_14"]),
            f"rsi14_{TREND_LOOKBACK}d_ago": _num(back.get("RSI_14")),
            "macd": _num(last["MACD"]), "macd_signal": _num(last["MACD_signal"]),
            "macd_hist": _num(last["MACD_hist"]),
            f"macd_hist_{TREND_LOOKBACK}d_ago": _num(back.get("MACD_hist")),
            "last_macd_signal_cross": _last_cross(df["MACD"] - df["MACD_signal"]),
        },
        "volatility": {
            "bb_upper": upper, "bb_mid": mid, "bb_lower": lower,
            # %B: 0 = at lower band, 1 = at upper band, >1 / <0 = outside the bands
            "bb_percent_b": _num((close - lower) / (upper - lower)) if None not in (close, upper, lower) and upper != lower else None,
            "bb_width_pct": _num(width.iloc[-1]),
            "bb_width_percentile_1y": width_pctile,
            "bollinger_squeeze": None if width_pctile is None else width_pctile <= SQUEEZE_PERCENTILE,
        },
    }


def generate_signal(df: pd.DataFrame, ticker: str, company: str | None,
                    news: SentimentAggregate | None = None, client: Groq | None = None) -> TradingSignal | None:
    """LLM Buy/Hold/Sell call over the technical context. News sentiment is optional secondary
    context (the requirement is a technical signal). None if there is no data or the LLM fails."""
    context = build_technical_context(df)
    if context is None:
        logger.error("No indicator data for %s; skipping signal", ticker)
        return None

    system = TRADING_SIGNAL_SYSTEM.format(min_sentences=MIN_SENTENCES, max_sentences=MAX_SENTENCES,
                                          min_families=MIN_INDICATOR_FAMILIES)
    user = TRADING_SIGNAL_USER.format(company=company or ticker, ticker=ticker,
                                      as_of=df.index[-1].date().isoformat(),
                                      indicators_json=json.dumps(context, indent=2))
    if news is not None and news.score is not None:
        user += TRADING_SIGNAL_NEWS_ADDENDUM.format(label=news.label, score=news.score, n_scored=news.n_scored)
    return call_llm_json(client or get_client(), system, user, TradingSignal,
                         reasoning_effort=SIGNAL_REASONING_EFFORT)


def setup_logging() -> None:
    """Console + file log, so validation failures are kept as evidence in outputs/."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(VALIDATION_LOG)])
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request is noise


if __name__ == "__main__":
    import sys

    from data_pipeline import DEFAULT_TICKER, run_pipeline

    setup_logging()
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    try:
        client = get_client()  # fail fast on a missing key, before spending time on data fetches
    except RuntimeError as exc:
        sys.exit(f"Config error: {exc}")
    df, news, summary = run_pipeline(ticker)
    company = summary.get("company")

    scored, failed = score_headlines(news, ticker, company, client)
    print(f"\nPer-headline sentiment ({MODEL}):")
    for s in scored:
        print(json.dumps(s.model_dump()))
    if failed:
        print(f"\nFailed ({len(failed)}): {failed}")
    sentiment = aggregate_sentiment(scored, len(failed))
    print("\nAggregate:\n" + sentiment.model_dump_json(indent=2))

    print("\nTechnical context sent to the LLM:\n" + json.dumps(build_technical_context(df), indent=2))
    print(f"\nRule-based momentum (Task 1A, for comparison): {summary.get('momentum')}")
    for label, news_ctx in [("technicals only", None), ("technicals + news", sentiment)]:
        signal = generate_signal(df, ticker, company, news_ctx, client)
        print(f"\nLLM signal ({label}):\n" + (signal.model_dump_json(indent=2) if signal else "FAILED - see log"))
