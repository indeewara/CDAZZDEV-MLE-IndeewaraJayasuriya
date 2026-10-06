"""Task 1A - financial data pipeline: OHLCV fetch, indicators, news, summary."""
import logging
import math
import re
import urllib.request
from collections import Counter
from difflib import SequenceMatcher

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---- Price history -------------------------------------------------------------------
DEFAULT_TICKER = "AAPL"
# 3y, not 2y: the 200-day SMA needs 200 bars of warm-up, so 3y leaves a full
# 2 years of rows where every indicator has a value.
HISTORY_YEARS = 3
MIN_ANALYSIS_YEARS = 2
TRADING_DAYS_PER_YEAR = 252
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# ---- Indicators (standard textbook parameters) --------------------------------------
SMA_SHORT, SMA_LONG = 50, 200
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_NUM_STD = 20, 2

# ---- News ----------------------------------------------------------------------------
MIN_HEADLINES = 10
NEWS_POOL_SIZE = 40           # collect this many candidates, then pick a diverse MIN_HEADLINES
MAX_PER_PUBLISHER = 2         # stops one outlet (e.g. a filings bot) dominating the sample
NEAR_DUP_SIMILARITY = 0.6     # difflib ratio on number-stripped titles; >= this = same story
HTTP_TIMEOUT_S = 10           # feedparser has no timeout of its own, so we fetch the bytes
HTTP_USER_AGENT = "Mozilla/5.0 (equity-research-assistant)"
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"

# ---- Summary / momentum --------------------------------------------------------------
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30   # Wilder's conventional bands
# 4 rules vote +1/-1 (RSI only votes at extremes). |score| >= 2 means the bullish/bearish
# votes outnumber the opposing ones by two, i.e. a clear majority rather than a split.
MOMENTUM_THRESHOLD = 2
SUMMARY_DECIMALS = 4
SUMMARY_INDICATORS = ["SMA_50", "SMA_200", "RSI_14", "MACD", "MACD_signal", "MACD_hist",
                      "BB_upper", "BB_mid", "BB_lower"]


# ======================================================================================
# 1. OHLCV
# ======================================================================================
def fetch_ohlcv(ticker: str = DEFAULT_TICKER, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Daily OHLCV for `ticker` over the last `years` years (relative, no hardcoded dates).

    Prices are split/dividend adjusted (auto_adjust=True) so returns and indicators are
    not distorted by corporate actions; they can differ slightly from Yahoo's raw display.
    Returns an empty DataFrame (and logs why) on any failure, so callers never
    see a raw yfinance/network exception.
    """
    try:
        df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=True,
                                       timeout=HTTP_TIMEOUT_S)
    except Exception as exc:  # yfinance raises many unrelated types (network, JSON, rate limit)
        logger.error("OHLCV fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    # reindex: a column yfinance stops sending becomes NaN instead of a KeyError
    df = (df if df is not None else pd.DataFrame()).reindex(columns=OHLCV_COLUMNS).dropna(subset=["Close"])
    if df.empty:
        logger.error("No usable OHLCV data for %s (bad ticker, delisted, or all-null Close)", ticker)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)  # drop tz so date math/plots are simple

    if len(df) < MIN_ANALYSIS_YEARS * TRADING_DAYS_PER_YEAR:
        logger.warning("%s has only %d rows, under %d years of history", ticker, len(df), MIN_ANALYSIS_YEARS)
    return df


# ======================================================================================
# 2. Indicators - pure pandas/numpy, no TA-Lib
# ======================================================================================
def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI with Wilder smoothing: seed = simple mean of the first `period` price changes,
    then avg = (prev_avg * (period - 1) + current) / period."""
    delta = close.diff().to_numpy()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return pd.Series(out, index=close.index)

    first_move = 1  # delta[0] is NaN (no previous close), so changes start at index 1
    avg_gain = gain[first_move : period + first_move].mean()
    avg_loss = loss[first_move : period + first_move].mean()
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        # no losses at all -> RSI is 100 by definition (avoids divide-by-zero)
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return pd.Series(out, index=close.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with SMA50/200, RSI14, MACD(12,26,9), Bollinger(20,2) columns."""
    out = df.copy()
    if out.empty or "Close" not in out:
        return out
    close = out["Close"]

    out["SMA_50"] = close.rolling(SMA_SHORT).mean()
    out["SMA_200"] = close.rolling(SMA_LONG).mean()
    out["RSI_14"] = rsi_wilder(close)

    # span=N -> alpha = 2/(N+1); adjust=False -> recursive EMA: ema_t = a*x_t + (1-a)*ema_{t-1}
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_signal"] = out["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # ddof=0: Bollinger's original definition uses population std (pandas defaults to 1)
    mid = close.rolling(BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW).std(ddof=0)
    out["BB_mid"], out["BB_upper"], out["BB_lower"] = mid, mid + BB_NUM_STD * std, mid - BB_NUM_STD * std
    return out


# ======================================================================================
# 3. News
# ======================================================================================
def _from_yfinance(ticker: str, n: int) -> list[dict]:
    """yfinance news. Handles both the new nested `content` schema and the legacy flat one."""
    items = yf.Ticker(ticker).get_news(count=n) or []
    out = []
    for item in items:
        c = item.get("content", item)
        provider = c.get("provider") or {}
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
        published = c.get("pubDate") or c.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = pd.Timestamp(published, unit="s", tz="UTC").isoformat()
        elif not isinstance(published, str):
            published = None  # unknown shape; keep the headline, drop the date
        out.append({"title": c.get("title"), "publisher": provider.get("displayName") or c.get("publisher"),
                    "published": published, "link": url, "source": "yfinance"})
    return out


def _from_rss(url: str, source: str) -> list[dict]:
    # Fetch ourselves with a timeout: feedparser.parse(url) can hang forever on a stalled socket.
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        feed = feedparser.parse(resp.read())
    out = []
    for e in feed.entries:
        publisher = (e.get("source") or {}).get("title")
        title = e.get("title", "")
        suffix = f" - {publisher}"  # Google appends " - Publisher" to every title
        if publisher and title.endswith(suffix):
            title = title[: -len(suffix)]
        published = e.get("published_parsed")
        out.append({"title": title, "publisher": publisher,
                    "published": pd.Timestamp(*published[:6], tz="UTC").isoformat() if published else None,
                    "link": e.get("link"), "source": source})
    return out


def _story_key(title: str) -> str:
    """Title with numbers/punctuation removed, so 'Insider X sells 46,389 shares' and
    'Insider Y sells 191,753 shares' compare as the same kind of story."""
    return re.sub(r"[^a-z ]+", "", (title or "").lower())


def select_diverse(pool: list[dict], n: int = MIN_HEADLINES) -> list[dict]:
    """Pick up to `n` headlines from `pool` (assumed newest first), skipping near-duplicate
    stories and capping each publisher. If that leaves fewer than `n`, top up with the
    newest leftovers - the count requirement beats perfect diversity."""
    picked, per_publisher = [], Counter()
    for h in pool:
        publisher = h["publisher"] or h["source"]
        if per_publisher[publisher] >= MAX_PER_PUBLISHER:
            continue
        key = _story_key(h["title"])
        if any(SequenceMatcher(None, key, _story_key(p["title"])).ratio() >= NEAR_DUP_SIMILARITY for p in picked):
            continue
        picked.append(h)
        per_publisher[publisher] += 1
        if len(picked) == n:
            return picked
    leftovers = [h for h in pool if h not in picked]
    return picked + leftovers[: n - len(picked)]


def fetch_news(ticker: str = DEFAULT_TICKER, n: int = MIN_HEADLINES) -> list[dict]:
    """`n` recent, varied headlines, newest first. Never raises.

    Sources in order: yfinance, Google News RSS, Yahoo RSS - stopping once NEWS_POOL_SIZE
    unique titles are collected. Google before Yahoo RSS: Yahoo's per-ticker feed is mostly
    general-market stories, which would pollute the per-ticker sentiment in Task 1B.
    """
    sources = [
        ("yfinance", lambda: _from_yfinance(ticker, NEWS_POOL_SIZE)),
        ("google_news_rss", lambda: _from_rss(GOOGLE_NEWS_RSS_URL.format(ticker=ticker), "google_news_rss")),
        ("yahoo_rss", lambda: _from_rss(YAHOO_RSS_URL.format(ticker=ticker), "yahoo_rss")),
    ]
    seen, pool = set(), []
    for name, fetch in sources:
        try:
            items = fetch()
        except Exception as exc:
            logger.warning("News source %s failed for %s: %s", name, ticker, exc)
            continue
        for item in items:
            key = (item["title"] or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                pool.append(item)
        logger.info("After %s: %d unique headlines", name, len(pool))
        if len(pool) >= NEWS_POOL_SIZE:
            break

    pool.sort(key=lambda h: h["published"] or "", reverse=True)  # ISO strings sort chronologically
    headlines = select_diverse(pool, n)
    if len(headlines) < n:
        logger.warning("Only %d headlines found for %s (wanted %d)", len(headlines), ticker, n)
    return headlines


# ======================================================================================
# 4. Summary
# ======================================================================================
def _num(x):
    """float, or None for NaN/inf/None/non-numeric - keeps the summary JSON-safe.
    (yfinance reports some P/E ratios as the string 'Infinity'.)"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, SUMMARY_DECIMALS) if math.isfinite(x) else None


def momentum_signal(row: pd.Series) -> dict:
    """Rule-based momentum from the latest indicator row. Each rule votes +1/-1;
    rules whose inputs are missing are skipped rather than guessed.

    Rules: price vs SMA50 (short trend), SMA50 vs SMA200 (long-term regime),
    MACD vs signal (momentum turning), RSI extremes (mean-reversion warning - contrarian,
    so overbought votes -1 even in an uptrend)."""
    votes = []
    price, sma50, sma200 = _num(row.get("Close")), _num(row.get("SMA_50")), _num(row.get("SMA_200"))
    macd, macd_sig, rsi = _num(row.get("MACD")), _num(row.get("MACD_signal")), _num(row.get("RSI_14"))

    if price is not None and sma50 is not None:
        votes.append((1, "price above SMA50") if price > sma50 else (-1, "price below SMA50"))
    if sma50 is not None and sma200 is not None:
        votes.append((1, "SMA50 above SMA200 (golden-cross regime)") if sma50 > sma200
                     else (-1, "SMA50 below SMA200 (death-cross regime)"))
    if macd is not None and macd_sig is not None:
        votes.append((1, "MACD above signal") if macd > macd_sig else (-1, "MACD below signal"))
    if rsi is not None:
        if rsi > RSI_OVERBOUGHT:
            votes.append((-1, f"RSI {rsi:.1f} overbought"))
        elif rsi < RSI_OVERSOLD:
            votes.append((1, f"RSI {rsi:.1f} oversold"))
        # between the bands: neutral zone, no vote

    score = sum(v for v, _ in votes)
    signal = "Bullish" if score >= MOMENTUM_THRESHOLD else "Bearish" if score <= -MOMENTUM_THRESHOLD else "Neutral"
    return {"signal": signal, "score": score, "reasons": [r for _, r in votes]}


def fetch_info(ticker: str = DEFAULT_TICKER) -> dict:
    """yfinance `info` dict, or {} on failure (it is slow and flaky)."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("info fetch failed for %s: %s", ticker, exc)
        return {}


def build_summary(df: pd.DataFrame, ticker: str = DEFAULT_TICKER, info: dict | None = None) -> dict:
    """Summary dict from an indicator frame (see add_indicators). Missing values become None.
    52w range and YTD are computed from the same adjusted series as the indicators (not
    taken from Yahoo's `info`), so all numbers are mutually consistent."""
    info = info or {}
    if df.empty or "Close" not in df:
        return {"ticker": ticker, "error": "no price data"}

    last = df.iloc[-1]
    last_year = df.tail(TRADING_DAYS_PER_YEAR)  # ~52 weeks of trading days
    no_data = pd.Series(dtype=float)  # missing High/Low column -> max/min is NaN -> None

    # YTD = last close vs last close of the previous calendar year (year taken from data, not hardcoded)
    year = df.index[-1].year
    prior = df.loc[df.index.year < year, "Close"]
    base = prior.iloc[-1] if not prior.empty else df.loc[df.index.year == year, "Close"].iloc[0]
    ytd = _num((last["Close"] / base - 1) * 100) if base else None

    pe = _num(info.get("trailingPE"))
    return {
        "ticker": ticker,
        "company": info.get("longName"),
        "currency": info.get("currency"),
        "as_of": df.index[-1].date().isoformat(),
        "current_price": _num(last["Close"]),
        "high_52w": _num(last_year.get("High", no_data).max()),
        "low_52w": _num(last_year.get("Low", no_data).min()),
        "pe_ratio": pe if pe and pe > 0 else None,  # negative/zero P/E is meaningless -> None
        "ytd_return_pct": ytd,
        "momentum": momentum_signal(last),
        "indicators": {col: _num(last.get(col)) for col in SUMMARY_INDICATORS},
    }


def run_pipeline(ticker: str = DEFAULT_TICKER) -> tuple[pd.DataFrame, list[dict], dict]:
    """Task 1A end to end: (indicator frame, headlines, summary)."""
    df = add_indicators(fetch_ohlcv(ticker))
    return df, fetch_news(ticker), build_summary(df, ticker, fetch_info(ticker))


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    data, news, summary = run_pipeline(ticker)
    span = f"{data.index[0].date()} -> {data.index[-1].date()}" if not data.empty else "no data"
    print(f"{ticker}: {len(data)} rows, {span}")
    print(f"\n{len(news)} headlines:")
    for h in news:
        print(f"  [{(h['published'] or '')[:10]}] {h['title']} ({h['publisher']}, via {h['source']})")
    print("\nSummary:\n" + json.dumps(summary, indent=2))
