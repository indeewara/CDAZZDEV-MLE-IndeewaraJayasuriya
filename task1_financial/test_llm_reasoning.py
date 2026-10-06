"""Offline tests for Task 1B: validation/retry, aggregation, and the signal, with a fake LLM client."""
import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from data_pipeline import add_indicators
from llm_reasoning import (aggregate_sentiment, build_technical_context, call_llm_json, generate_signal,
                           score_headlines)
from schemas import HeadlineSentiment, TradingSignal, count_sentences


class FakeClient:
    """Returns the queued raw strings as successive LLM replies and records each request."""
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs["messages"])
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


GOOD = '{"headline": "H", "sentiment": "Positive", "confidence": 0.8, "brief_reason": "Beat."}'


def test_invalid_reply_is_logged_and_repaired(caplog):
    bad = '{"headline": "H", "sentiment": "bullish", "confidence": 1.7, "brief_reason": "x"}'
    client = FakeClient([bad, GOOD])
    with caplog.at_level(logging.WARNING):
        out = call_llm_json(client, "sys", "user", HeadlineSentiment)
    assert out.sentiment == "positive" and out.confidence == 0.8  # case normalised
    assert "Validation failed" in caplog.text
    assert "failed validation" in client.requests[1][-1]["content"]  # errors sent back to the model


def test_gives_up_gracefully_after_max_attempts(caplog):
    client = FakeClient(["not json", "{}", '{"sentiment": "positive"}'])
    with caplog.at_level(logging.ERROR):
        assert call_llm_json(client, "sys", "user", HeadlineSentiment) is None
    assert "Giving up" in caplog.text


def test_malformed_envelope_and_api_errors_are_handled(caplog):
    import groq
    import httpx

    class Flaky(FakeClient):
        """Queue items: an Exception is raised, None is a reply with no choices, a str is content."""
        def _create(self, **kwargs):
            reply = self.replies[0]
            if isinstance(reply, Exception):
                self.replies.pop(0)
                self.requests.append(kwargs["messages"])
                raise reply
            if reply is None:
                self.replies.pop(0)
                self.requests.append(kwargs["messages"])
                return SimpleNamespace(choices=[])
            return super()._create(**kwargs)

    timeout = groq.APITimeoutError(request=httpx.Request("POST", "https://x"))
    client = Flaky([timeout, None, GOOD])
    with caplog.at_level(logging.WARNING):
        out = call_llm_json(client, "sys", "user", HeadlineSentiment)
    assert out is not None and len(client.requests) == 3
    assert "API error" in caplog.text and "Validation failed" in caplog.text


def test_failed_headline_reported_and_paraphrase_restored():
    client = FakeClient([GOOD, "x", "x", "x"])
    scored, failed = score_headlines([{"title": "Real headline"}, {"title": "Broken"}], "T", "Co", client)
    assert [s.headline for s in scored] == ["Real headline"]  # model said "H"; original kept
    assert failed == ["Broken"]


def _s(sentiment, conf):
    return HeadlineSentiment(headline="h", sentiment=sentiment, confidence=conf, brief_reason="r")


def test_aggregate_is_confidence_weighted_and_neutrals_dilute():
    agg = aggregate_sentiment([_s("positive", 0.9), _s("negative", 0.3), _s("neutral", 0.6)], n_failed=1)
    assert abs(agg.score - (0.9 - 0.3) / 1.8) < 1e-4  # = 0.3333
    assert agg.label == "positive" and agg.counts == {"positive": 1, "neutral": 1, "negative": 1}
    assert agg.n_failed == 1


def test_aggregate_neutral_band_and_empty():
    assert aggregate_sentiment([_s("positive", 0.1), _s("neutral", 0.9)]).label == "neutral"  # 0.1/1.0 < 0.15
    assert aggregate_sentiment([_s("positive", 0.2), _s("neutral", 0.9)]).label == "positive"  # 0.2/1.1 = 0.18
    empty = aggregate_sentiment([], n_failed=10)
    assert empty.score is None and empty.label == "unknown"


# ---- Trading signal

GOOD_JUSTIFICATION = (
    "Price holds 3.3% above a rising SMA 50 and the 50-day sits well above the 200-day, so the primary trend is up. "
    "However, MACD has crossed below its signal line while RSI sits at a neutral 54, which reads as momentum "
    "cooling inside the uptrend rather than a reversal. "
    "Price is mid-band on the Bollinger Bands, so there is no stretched move to fade. "
    "The trend outweighs the softening momentum, so the call is Hold until MACD turns back up."
)


def _signal_json(**overrides):
    body = {"signal": "hold", "confidence": 0.6, "key_conflict": "Trend up, momentum fading.",
            "justification": GOOD_JUSTIFICATION}
    return json.dumps({**body, **overrides})


def test_sentence_counter_ignores_decimals():
    assert count_sentences("Price is 3.3% above SMA. RSI is 54.2 today. Done.") == 3


def test_signal_schema_accepts_good_and_normalises_case():
    s = TradingSignal.model_validate_json(_signal_json())
    assert s.signal == "Hold"


@pytest.mark.parametrize("bad, why", [
    ({"justification": "RSI is 54. MACD is 3.5."}, "sentences"),                     # too short
    ({"justification": " ".join(["SMA, RSI and MACD agree."] * 6)}, "sentences"),    # too long
    ({"justification": "RSI is 54. RSI was 50. RSI rises. RSI is fine."}, "families"),  # one indicator only
    ({"signal": "Strong Buy"}, "signal"),
    ({"confidence": 1.4}, "confidence"),
])
def test_signal_schema_rejects(bad, why):
    with pytest.raises(ValidationError, match=why):  # fails for the right reason
        TradingSignal.model_validate_json(_signal_json(**bad))


def _uptrend_frame():
    t = np.arange(400)
    close = 100 + 0.002 * t**2 + np.sin(t / 5)
    idx = pd.bdate_range("2025-01-01", periods=len(t))
    return add_indicators(pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close}, index=idx))


def test_technical_context_derives_relationships():
    ctx = build_technical_context(_uptrend_frame())
    assert ctx["trend"]["price_vs_sma200_pct"] > 0 and ctx["trend"]["sma50_vs_sma200_pct"] > 0
    assert ctx["trend"]["last_sma50_sma200_cross"] is None  # SMA50 above SMA200 since it first existed
    assert 0 <= ctx["volatility"]["bb_width_percentile_1y"] <= 100
    json.dumps(ctx, allow_nan=False)  # JSON-safe: no NaN/inf reaches the prompt


def test_last_cross_detects_recent_golden_cross():
    close = np.r_[np.linspace(200, 100, 300), np.linspace(100, 220, 150)]  # fall, then strong rally
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    ctx = build_technical_context(add_indicators(pd.DataFrame({"High": close, "Low": close, "Close": close}, index=idx)))
    cross = ctx["trend"]["last_sma50_sma200_cross"]
    assert cross["direction"] == "bullish" and 0 < cross["days_ago"] < 150


def test_generate_signal_end_to_end_with_repair_and_news():
    client = FakeClient([_signal_json(justification="Too short."), _signal_json()])
    agg = aggregate_sentiment([_s("negative", 0.8)])
    sig = generate_signal(_uptrend_frame(), "T", "Co", news=agg, client=client)
    assert sig.signal == "Hold"
    first_user_msg = client.requests[0][1]["content"]
    assert '"price_vs_sma50_pct"' in first_user_msg and "news sentiment" in first_user_msg
    assert "failed validation" in client.requests[1][-1]["content"]


def test_generate_signal_skips_llm_when_no_data():
    client = FakeClient([])
    assert generate_signal(pd.DataFrame(), "T", None, client=client) is None
    assert client.requests == []
