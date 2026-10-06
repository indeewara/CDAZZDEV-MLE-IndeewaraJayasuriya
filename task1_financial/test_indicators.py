"""Checks indicators against known values. Run: pytest task1_financial"""
import numpy as np
import pandas as pd

import statistics

from data_pipeline import add_indicators, build_summary, momentum_signal, rsi_wilder, select_diverse

# Wilder's worked RSI example (StockCharts "RSI" reference data)
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
    45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
]
WILDER_RSI = {14: 70.53, 15: 66.32, 16: 66.55, 17: 69.41, 18: 66.36, 19: 57.97}


def test_rsi_matches_wilder_reference():
    rsi = rsi_wilder(pd.Series(WILDER_CLOSES))
    assert rsi.iloc[:14].isna().all()
    for i, expected in WILDER_RSI.items():
        # 0.2 tolerance: reference was computed from unrounded prices, ours from the 2dp closes
        assert abs(rsi.iloc[i] - expected) < 0.2, (i, rsi.iloc[i], expected)
    assert abs(rsi.iloc[14] - 70.4641) < 1e-3  # hand calc: gain 3.34/14, loss 1.40/14


def test_rsi_all_gains_is_100():
    assert rsi_wilder(pd.Series(np.arange(1.0, 40.0))).iloc[-1] == 100.0


def _frame(close):
    return pd.DataFrame({"Close": close})


def test_constant_price_has_zero_macd_and_collapsed_bands():
    out = add_indicators(_frame([100.0] * 300))
    assert out["MACD"].abs().max() < 1e-9
    assert (out["BB_upper"] - out["BB_lower"]).dropna().abs().max() < 1e-9
    assert out["SMA_200"].iloc[-1] == 100.0 and out["SMA_200"].iloc[:199].isna().all()


def test_sma_and_bollinger_population_std():
    out = add_indicators(_frame(np.arange(1.0, 301.0)))
    assert out["SMA_50"].iloc[-1] == np.arange(251.0, 301.0).mean()
    window = np.arange(281.0, 301.0)  # last 20 closes
    assert abs(out["BB_upper"].iloc[-1] - (window.mean() + 2 * window.std())) < 1e-9


def test_empty_frame_does_not_raise():
    assert add_indicators(pd.DataFrame()).empty
    assert build_summary(pd.DataFrame())["error"] == "no price data"


def _priced(close, start="2024-06-03"):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close))
    return add_indicators(pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close}, index=idx))


def test_momentum_uptrend_is_bullish_downtrend_is_bearish():
    # accelerating trends: a straight line makes MACD == signal (degenerate tie)
    t = np.arange(300)
    up = momentum_signal(_priced(100 + 0.002 * t**2).iloc[-1])
    down = momentum_signal(_priced(300 - 0.002 * t**2).iloc[-1])
    assert up["signal"] == "Bullish", up
    assert down["signal"] == "Bearish", down


def test_momentum_skips_missing_inputs():
    m = momentum_signal(_priced(np.linspace(100, 110, 30)).iloc[-1])  # too short for SMA50/200
    assert not any("SMA" in r for r in m["reasons"])


def test_missing_data_never_raises(monkeypatch):
    import data_pipeline as dp

    class FakeTicker:
        def __init__(self, hist): self.hist = hist
        def history(self, **_): return self.hist

    # no Open/Volume columns, as if yfinance dropped them
    no_volume = pd.DataFrame({"High": 2.0, "Low": 0.5, "Close": np.linspace(100, 120, 300)},
                             index=pd.bdate_range("2025-01-01", periods=300))
    for hist in [None, no_volume, no_volume.assign(Close=np.nan)]:
        monkeypatch.setattr(dp.yf, "Ticker", lambda *_ , h=hist: FakeTicker(h))
        build_summary(add_indicators(dp.fetch_ohlcv("X")))  # must not raise

    base = _priced(np.linspace(100, 120, 300))
    assert build_summary(base, "X", {"trailingPE": "Infinity"})["pe_ratio"] is None
    assert build_summary(base.drop(columns=["High", "Low"]))["high_52w"] is None
    assert build_summary(pd.DataFrame({"Open": [1.0]}))["error"] == "no price data"


def test_summary_fields_ytd_and_bad_pe():
    df = _priced(np.linspace(100, 200, 300))  # 2024-06-03 .. mid-2025
    s = build_summary(df, "TEST", {"trailingPE": -5, "longName": "Test Co"})
    prior_close = df.loc[df.index.year < df.index[-1].year, "Close"].iloc[-1]
    assert abs(s["ytd_return_pct"] - (200 / prior_close - 1) * 100) < 1e-3
    assert s["pe_ratio"] is None  # negative P/E rejected
    assert s["high_52w"] == 201.0 and s["current_price"] == 200.0
    assert s["indicators"]["SMA_200"] is not None


# ---- Independent cross-check: plain-Python loops written straight from the textbook
# definitions (no pandas ewm/rolling/std), compared on a realistic random-walk series.

def _ema_textbook(xs, n):
    """EMA_t = a*x_t + (1-a)*EMA_{t-1}, a = 2/(n+1), seeded with the first value."""
    a, out = 2 / (n + 1), [xs[0]]
    for x in xs[1:]:
        out.append(a * x + (1 - a) * out[-1])
    return out


def test_macd_and_bollinger_match_textbook_loops():
    rng = np.random.default_rng(42)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400))))  # ~1.5% daily vol
    out = add_indicators(_frame(closes))

    macd = [f - s for f, s in zip(_ema_textbook(closes, 12), _ema_textbook(closes, 26))]
    signal = _ema_textbook(macd, 9)
    assert np.allclose(out["MACD"], macd, atol=1e-9)
    assert np.allclose(out["MACD_signal"], signal, atol=1e-9)
    assert np.allclose(out["MACD_hist"], [m - s for m, s in zip(macd, signal)], atol=1e-9)

    for i in range(19, len(closes)):
        window = closes[i - 19 : i + 1]
        mid, sd = sum(window) / 20, statistics.pstdev(window)  # population std, as Bollinger defined
        assert abs(out["BB_mid"].iloc[i] - mid) < 1e-9
        assert abs(out["BB_upper"].iloc[i] - (mid + 2 * sd)) < 1e-9
        assert abs(out["BB_lower"].iloc[i] - (mid - 2 * sd)) < 1e-9

    for i in range(199, len(closes)):
        assert abs(out["SMA_200"].iloc[i] - sum(closes[i - 199 : i + 1]) / 200) < 1e-9
        assert abs(out["SMA_50"].iloc[i] - sum(closes[i - 49 : i + 1]) / 50) < 1e-9


def test_rsi_bounded_on_random_walk():
    rng = np.random.default_rng(7)
    rsi = rsi_wilder(pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all() and len(rsi) == 500 - 14


# ---- News diversity

def _h(title, publisher, day):
    return {"title": title, "publisher": publisher, "published": f"2026-10-{day:02d}", "source": "t"}


def test_select_diverse_caps_publisher_and_drops_near_duplicates():
    insider = [_h(f"Apple (NASDAQ:AAPL) Stock: Insider Person{i} Sells {1000 * i:,} Shares", "MarketBeat", 9)
               for i in range(1, 9)]
    other = [_h(t, p, 8) for t, p in [
        ("Apple unveils foldable iPhone at October event", "Reuters"),
        ("Why Apple's services margin keeps expanding", "Barron's"),
        ("Apple faces EU fine over App Store rules", "Bloomberg"),
        ("Analysts raise Apple price target ahead of earnings", "CNBC"),
        ("Apple supplier Foxconn reports record revenue", "Nikkei"),
        ("Is Apple stock a buy after its 20% YTD rally?", "Motley Fool"),
        ("Apple CEO outlines AI roadmap in interview", "The Verge"),
        ("Apple dividend: how many shares for $10,000 a year", "Yahoo Finance"),
        ("Treasury yields pressure big tech including Apple", "WSJ"),
    ]]
    picked = select_diverse(insider + other, n=10)
    assert len(picked) == 10
    assert sum(p["publisher"] == "MarketBeat" for p in picked) == 1  # 8 near-identical filings -> 1


def test_select_diverse_tops_up_when_pool_is_homogeneous():
    same = [_h(f"Insider sells {i} shares", "Bot", 1) for i in range(15)]
    assert len(select_diverse(same, n=10)) == 10  # count requirement still met
