"""The five research tools. Each returns a compact, JSON-serialisable dict; failures return {"error": ...}
(never raise), so the agent can observe the failure and try something else.

Price, indicator and news logic is reused from Task 1 (task1_financial/data_pipeline.py), which is already tested.
Outputs are deliberately small: every observation is re-sent to the LLM on later turns, and the Groq free tier
allows 8,000 tokens per minute.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from agent_prompts import BATCH_SENTIMENT_SYSTEM, BATCH_SENTIMENT_USER
from tracing import estimate_tokens, groq_client, limiter, traced

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task1_financial"))
from data_pipeline import add_indicators, build_summary, fetch_info, fetch_news, fetch_ohlcv  # noqa: E402
from llm_reasoning import aggregate_sentiment  # noqa: E402
from schemas import HeadlineSentiment  # noqa: E402  (Task 1's schema)

TRADING_DAYS = 252
PERIOD_YEARS = {"6mo": 1, "1y": 1, "2y": 2, "3y": 3, "5y": 5}  # >= 1y fetched so the 200-day SMA has history
RECENT_ROWS = 5                  # OHLCV rows returned: enough to see the latest moves without flooding the context
HEDGE_HORIZON_DAYS = 90          # the question asks about the next 90 days
MAX_HEADLINES = 15
MAX_SEARCH_RESULTS = 8
SNIPPET_CHARS = 250
SENTIMENT_MODEL = "openai/gpt-oss-20b"   # separate Groq quota from the agent's model, and plenty for labelling


def _round(x, nd=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def get_price_data(ticker: str, period: str = "1y") -> dict:
    """Daily OHLCV price history for a stock with computed technical indicators (SMA 50/200, RSI 14, MACD,
    Bollinger Bands), plus current price, 52-week range, P/E, YTD return and a momentum signal.
    period: one of 6mo, 1y, 2y, 3y, 5y."""
    ticker = ticker.strip().upper()
    if period not in PERIOD_YEARS:
        return {"error": f"unsupported period {period!r}; use one of {list(PERIOD_YEARS)}"}
    df = add_indicators(fetch_ohlcv(ticker, PERIOD_YEARS[period]))
    if df.empty:
        return {"error": f"no price data for {ticker} - check the ticker symbol"}
    summary = build_summary(df, ticker, fetch_info(ticker))
    window = df.tail(TRADING_DAYS // 2) if period == "6mo" else df
    cols = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "SMA_200", "RSI_14", "MACD_hist"]
    recent = [{"date": d.date().isoformat(), **{c: _round(r[c]) for c in cols if c in r}}
              for d, r in df.tail(RECENT_ROWS).iterrows()]
    return {
        "ticker": ticker, "company": summary.get("company"), "period": period, "rows": len(window),
        "period_return_pct": _round((window["Close"].iloc[-1] / window["Close"].iloc[0] - 1) * 100),
        "current_price": _round(summary["current_price"]), "high_52w": _round(summary["high_52w"]),
        "low_52w": _round(summary["low_52w"]), "pe_ratio": _round(summary["pe_ratio"], 1),
        "ytd_return_pct": _round(summary["ytd_return_pct"]),
        "momentum": summary["momentum"], "latest_indicators": {k: _round(v) for k, v in summary["indicators"].items()},
        "recent_ohlcv": recent,
    }


def calculate_volatility(ticker: str, window: int = 30) -> dict:
    """Annualised historical volatility of daily log returns over a trailing window of trading days, compared with
    the past year, plus the expected 1-sigma price move over the next 90 days and the 1-year max drawdown."""
    ticker = ticker.strip().upper()
    if not 5 <= window <= TRADING_DAYS:
        return {"error": f"window must be between 5 and {TRADING_DAYS} trading days, got {window}"}
    df = fetch_ohlcv(ticker, 2)
    if len(df) < window + TRADING_DAYS // 2:
        return {"error": f"not enough price history for {ticker} to compute a {window}-day volatility"}
    close = df["Close"]
    rolling = np.log(close).diff().rolling(window).std() * math.sqrt(TRADING_DAYS)
    year = rolling.tail(TRADING_DAYS).dropna()
    current = float(rolling.iloc[-1])
    last_year_close = close.tail(TRADING_DAYS)
    drawdown = (last_year_close / last_year_close.cummax() - 1).min()
    price = float(close.iloc[-1])
    move = current * math.sqrt(HEDGE_HORIZON_DAYS / TRADING_DAYS)   # 1-sigma move over 90 trading days
    return {
        "ticker": ticker, "window_days": window,
        "annualised_vol_pct": _round(current * 100),
        "one_year_avg_vol_pct": _round(year.mean() * 100),
        "one_year_range_vol_pct": [_round(year.min() * 100), _round(year.max() * 100)],
        "percentile_vs_one_year": _round((year <= current).mean() * 100, 0),
        "expected_90d_move_pct_1sigma": _round(move * 100),
        "current_price": _round(price),
        # 1-sigma price band over the hedge horizon - computed here so the LLM never does option-strike arithmetic
        "price_90d_minus_1sigma": _round(price * (1 - move)),
        "price_90d_plus_1sigma": _round(price * (1 + move)),
        "max_drawdown_1y_pct": _round(drawdown * 100),
    }


def get_news(ticker: str, n: int = 10) -> dict:
    """Recent news headlines for a stock (title, publisher, date), newest first."""
    ticker = ticker.strip().upper()
    n = max(1, min(int(n), 20))
    headlines = fetch_news(ticker, n)
    if not headlines:
        return {"error": f"no headlines found for {ticker}"}
    return {"ticker": ticker, "count": len(headlines),
            "headlines": [{"title": h["title"], "publisher": h.get("publisher"),
                           "date": (h.get("published") or "")[:10]} for h in headlines]}


def llm_sentiment(headlines: list[str]) -> dict:
    """Scores a list of news headline strings with an LLM (positive / negative / neutral with confidence) and
    returns a confidence-weighted aggregate sentiment score from -1 (negative) to +1 (positive)."""
    headlines = [h.strip() for h in headlines if isinstance(h, str) and h.strip()][:MAX_HEADLINES]
    if not headlines:
        return {"error": "no headlines given - pass a list of headline strings"}
    client = groq_client()
    user = BATCH_SENTIMENT_USER.format(headlines="\n".join(f"- {h}" for h in headlines))
    lim = limiter(SENTIMENT_MODEL)
    lim.wait(estimate_tokens(BATCH_SENTIMENT_SYSTEM + user))
    resp = client.chat.completions.create(
        model=SENTIMENT_MODEL, temperature=0, reasoning_effort="low", response_format={"type": "json_object"},
        messages=[{"role": "system", "content": BATCH_SENTIMENT_SYSTEM}, {"role": "user", "content": user}])
    lim.record(resp.usage.total_tokens)
    try:
        items = json.loads(resp.choices[0].message.content or "{}").get("results", [])
    except json.JSONDecodeError:
        return {"error": "sentiment model returned invalid JSON"}
    scored, failed = [], 0
    for item in items if isinstance(items, list) else []:
        try:
            scored.append(HeadlineSentiment.model_validate(item))   # validated before use, as in Task 1
        except ValidationError:
            failed += 1
    failed += max(0, len(headlines) - len(scored) - failed)
    if not scored:
        return {"error": "no headline could be scored"}
    agg = aggregate_sentiment(scored, failed)
    return {"aggregate": agg.model_dump(),
            "per_headline": [{"headline": s.headline[:90], "sentiment": s.sentiment, "confidence": s.confidence}
                             for s in scored]}


def web_search(query: str, max_results: int = 5) -> dict:
    """Web search (DuckDuckGo) for analyst commentary, price targets, upcoming events and risks not covered by
    the headlines. Returns titles, snippets and URLs."""
    from ddgs import DDGS
    max_results = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    results = DDGS().text(query, max_results=max_results) or []
    if not results:
        return {"error": f"no web results for {query!r}"}
    return {"query": query, "results": [{"title": r.get("title"), "snippet": (r.get("body") or "")[:SNIPPET_CHARS],
                                         "url": r.get("href")} for r in results]}


TOOL_FUNCTIONS = [get_price_data, calculate_volatility, get_news, llm_sentiment, web_search]
TOOLS = {fn.__name__: StructuredTool.from_function(traced(fn), name=fn.__name__, description=fn.__doc__)
         for fn in TOOL_FUNCTIONS}
