"""Bonus - one-page equity research brief: Markdown, rendered to a styled HTML page with an
embedded matplotlib chart. Combines the Task 1A pipeline and the Task 1B LLM outputs."""
import base64
import html
import io
import logging
from datetime import date
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")  # headless: render to PNG without a display
import matplotlib.pyplot as plt
import pandas as pd

from data_pipeline import TRADING_DAYS_PER_YEAR
from llm_reasoning import MODEL, build_technical_context
from schemas import HeadlineSentiment, SentimentAggregate, TradingSignal

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHART_DAYS = TRADING_DAYS_PER_YEAR  # chart the last ~12 months
TOP_HEADLINES = 3
CHART_DPI = 150

# Chart colours: first three slots of a CVD-validated categorical palette (they pass
# all-pairs colour-blind separation), plus neutral greys for bands and reference lines.
SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
BAND_FILL, REFERENCE, GRID, INK_MUTED = "#e9e8e4", "#8a8984", "#ecebe7", "#52514e"

DISCLAIMER = (
    "This brief is generated automatically by a software pipeline and a large language model for "
    "educational and assessment purposes only. It is not investment advice, a solicitation, or a "
    "recommendation to buy or sell any security. Technical indicators describe past prices and do not "
    "predict future returns; LLM output can be wrong, incomplete, or inconsistent between runs; news "
    "sentiment is inferred from headlines alone. Data may be delayed or inaccurate. Past performance is "
    "not indicative of future results. Consult a qualified financial adviser before making any "
    "investment decision."
)


# ---- helpers -------------------------------------------------------------------------
def _esc(text) -> str:
    """Escape external text (headlines, LLM output) for Markdown-then-HTML: no raw HTML
    injection, and no accidental *emphasis* or broken table cells."""
    s = html.escape(str(text), quote=False)
    for ch in "\\`*_[]|#":
        s = s.replace(ch, "\\" + ch)
    return s


def _fmt(x, spec=",.2f", suffix="", missing="n/a") -> str:
    return missing if x is None else f"{x:{spec}}{suffix}"


def _signed(x, spec=".1f", suffix="%") -> str:
    return "n/a" if x is None else f"{x:+{spec}}{suffix}"


def _ordinal(x) -> str:
    if x is None:
        return "n/a"
    n = round(x)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def top_headlines(scored: list[HeadlineSentiment], k: int = TOP_HEADLINES) -> list[HeadlineSentiment]:
    """Most market-moving first: directional (positive/negative) before neutral, then by confidence."""
    return sorted(scored, key=lambda s: (s.sentiment != "neutral", s.confidence), reverse=True)[:k]


# ---- chart ---------------------------------------------------------------------------
def render_chart(df: pd.DataFrame, ticker: str) -> bytes:
    """Three stacked panels sharing one time axis (never a dual y-axis):
    price + SMA 50/200 + Bollinger band, RSI 14 with 30/70 bands, MACD with histogram."""
    d = df.tail(CHART_DAYS)
    fig, (ax_px, ax_rsi, ax_macd) = plt.subplots(
        3, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.2]})

    ax_px.fill_between(d.index, d["BB_lower"], d["BB_upper"], color=BAND_FILL, lw=0, label="Bollinger (20, 2σ)")
    ax_px.plot(d.index, d["Close"], color=SERIES_1, lw=1.6, label="Close")
    ax_px.plot(d.index, d["SMA_50"], color=SERIES_2, lw=1.4, label="SMA 50")
    ax_px.plot(d.index, d["SMA_200"], color=SERIES_3, lw=1.4, label="SMA 200")
    ax_px.set_ylabel("Price")
    # legends sit above each panel, so they can never cover the data
    ax_px.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=4, frameon=False, fontsize=8)

    ax_rsi.axhspan(30, 70, color=BAND_FILL, lw=0)
    for level in (30, 70):
        ax_rsi.axhline(level, color=REFERENCE, lw=0.8, ls="--")
    ax_rsi.plot(d.index, d["RSI_14"], color=SERIES_1, lw=1.4)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_yticks([30, 70])
    ax_rsi.set_ylabel("RSI 14")

    ax_macd.bar(d.index, d["MACD_hist"], color=REFERENCE, width=1.0, label="Histogram")
    ax_macd.plot(d.index, d["MACD"], color=SERIES_1, lw=1.4, label="MACD")
    ax_macd.plot(d.index, d["MACD_signal"], color=SERIES_2, lw=1.4, label="Signal")
    ax_macd.axhline(0, color=REFERENCE, lw=0.8)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3, frameon=False, fontsize=8)

    for ax in (ax_px, ax_rsi, ax_macd):  # recessive axes and grid
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        ax.yaxis.label.set_color(INK_MUTED)
        ax.yaxis.label.set_fontsize(9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    ax_px.set_title(f"{ticker} - last {CHART_DAYS} trading days", loc="left", fontsize=10, color=INK_MUTED)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI)
    plt.close(fig)
    return buf.getvalue()


# ---- markdown ------------------------------------------------------------------------
def build_markdown(summary: dict, df: pd.DataFrame, scored: list[HeadlineSentiment],
                   sentiment: SentimentAggregate, signal: TradingSignal | None,
                   news: list[dict], chart_src: str) -> str:
    """The brief as Markdown. `chart_src` is a file path (for the .md) or a data URI (for HTML)."""
    ticker = summary["ticker"]
    company = summary.get("company") or ticker
    ccy = summary.get("currency") or ""
    ctx = build_technical_context(df) or {}
    trend, mom, vol = ctx.get("trend", {}), ctx.get("momentum", {}), ctx.get("volatility", {})
    momentum = summary.get("momentum", {})
    links = {h["title"]: h.get("link") for h in news}

    hi, lo, px = summary.get("high_52w"), summary.get("low_52w"), summary.get("current_price")
    range_pos = (px - lo) / (hi - lo) * 100 if None not in (hi, lo, px) and hi != lo else None
    cross = trend.get("last_sma50_sma200_cross")
    macd_cross = mom.get("last_macd_signal_cross")
    hist_now, hist_then = mom.get("macd_hist"), mom.get("macd_hist_10d_ago")
    hist_dir = ("rising" if hist_now > hist_then else "falling") if None not in (hist_now, hist_then) else "n/a"
    call = signal.signal.upper() if signal else "UNAVAILABLE"

    lines = [
        f"# {_esc(company)} ({ticker})",
        f"Equity research brief · data as of {summary.get('as_of')} · generated {date.today().isoformat()}",
        "",
        f'<div class="call call-{call.lower()}"><span>LLM recommendation</span><strong>{call}</strong>'
        + (f"<em>confidence {signal.confidence:.0%}</em>" if signal else "") + "</div>",
        "",
        "## Company snapshot",
        "",
        "| Price | 52-week range | Position in range | P/E (trailing) | YTD return |",
        "|---|---|---|---|---|",
        f"| {_fmt(px)} {ccy} | {_fmt(lo)} - {_fmt(hi)} | {_fmt(range_pos, '.0f', '%')} | "
        f"{_fmt(summary.get('pe_ratio'), '.1f')} | {_signed(summary.get('ytd_return_pct'))} |",
        "",
        "## Technical outlook",
        "",
        f"![{ticker} price, moving averages, Bollinger Bands, RSI and MACD]({chart_src})",
        "",
        "| | Reading | Detail |",
        "|---|---|---|",
        f"| **Trend** | Price {_signed(trend.get('price_vs_sma50_pct'))} vs SMA 50, "
        f"{_signed(trend.get('price_vs_sma200_pct'))} vs SMA 200 | "
        + (f"Last SMA 50/200 cross: {cross['direction']} ({cross['days_ago']} days ago)" if cross else "No SMA cross in range")
        + " |",
        f"| **Momentum** | RSI {_fmt(mom.get('rsi14'), '.1f')} (10d ago {_fmt(mom.get('rsi14_10d_ago'), '.1f')}) | "
        f"MACD histogram {_fmt(hist_now, '+.2f')}, {hist_dir}"
        + (f"; {macd_cross['direction']} signal cross {macd_cross['days_ago']}d ago" if macd_cross else "") + " |",
        f"| **Volatility** | Bollinger %B {_fmt(vol.get('bb_percent_b'), '.2f')} | "
        f"Band width at {_ordinal(vol.get('bb_width_percentile_1y'))} percentile of 1y"
        + (" (squeeze)" if vol.get("bollinger_squeeze") else "") + " |",
        f"| **Rule-based momentum** | {momentum.get('signal', 'n/a')} (score {momentum.get('score', 'n/a')}) | "
        f"{_esc('; '.join(momentum.get('reasons', [])) or 'n/a')} |",
        "",
        "## News sentiment",
        "",
        f"**{sentiment.label.capitalize()}** · score {_fmt(sentiment.score, '+.2f')} on a -1 to +1 scale · "
        f"{sentiment.counts.get('positive', 0)} positive, {sentiment.counts.get('neutral', 0)} neutral, "
        f"{sentiment.counts.get('negative', 0)} negative of {sentiment.n_scored} headlines"
        + (f" ({sentiment.n_failed} could not be scored)" if sentiment.n_failed else ""),
        "",
        "| Top headlines | Sentiment | Why |",
        "|---|---|---|",
    ]
    for s in top_headlines(scored):
        link = links.get(s.headline)
        title = f"[{_esc(s.headline)}]({link})" if link and link.startswith(("http://", "https://")) else _esc(s.headline)
        lines.append(f"| {title} | {s.sentiment} ({s.confidence:.0%}) | {_esc(s.brief_reason)} |")

    lines += ["", "## Recommendation and reasoning", ""]
    if signal:
        lines += [f"**{signal.signal}** (confidence {signal.confidence:.0%}). {_esc(signal.justification)}", "",
                  f"*Key conflict:* {_esc(signal.key_conflict)}", ""]
    else:
        lines += ["The LLM recommendation could not be produced (validation failed after retries; "
                  "see outputs/llm_validation.log).", ""]
    lines += [f"<small>Signal and sentiment by {MODEL} via Groq, reasoning over indicators computed "
              "from first principles; prices are split/dividend adjusted.</small>", "",
              "## Risk disclaimer", "", f'<p class="disclaimer">{DISCLAIMER}</p>']
    return "\n".join(lines)


# ---- html ----------------------------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --ink:#0b0b0b; --ink-2:#52514e; --line:#e2e1dc; --paper:#fcfcfb; --accent:#2a78d6;
           --buy:#0f7a3d; --hold:#9a6400; --sell:#b42318; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f1f0ec; color:var(--ink);
          font:13px/1.5 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  main {{ max-width:860px; margin:24px auto; padding:28px 36px; background:var(--paper);
          border:1px solid var(--line); border-radius:6px; }}
  h1 {{ font-size:22px; margin:0 0 2px; letter-spacing:-.01em; }}
  h1 + p {{ color:var(--ink-2); margin:0 0 14px; font-size:12px; }}
  h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--accent);
        border-bottom:1px solid var(--line); padding-bottom:4px; margin:18px 0 8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin:4px 0; }}
  th, td {{ text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ color:var(--ink-2); font-weight:600; background:#f6f5f2; }}
  img {{ max-width:100%; display:block; margin:4px 0 6px; }}
  a {{ color:var(--accent); text-decoration:none; }}
  .call {{ display:flex; align-items:baseline; gap:12px; padding:10px 14px; border-radius:6px;
           border:1px solid var(--line); border-left:6px solid var(--hold); background:#fff; }}
  .call span {{ color:var(--ink-2); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
  .call strong {{ font-size:20px; letter-spacing:.04em; }}
  .call em {{ color:var(--ink-2); font-style:normal; }}
  .call-buy {{ border-left-color:var(--buy); }} .call-buy strong {{ color:var(--buy); }}
  .call-hold strong {{ color:var(--hold); }}
  .call-sell {{ border-left-color:var(--sell); }} .call-sell strong {{ color:var(--sell); }}
  small {{ color:var(--ink-2); }}
  .disclaimer {{ font-size:10.5px; color:var(--ink-2); background:#f6f5f2; padding:8px 10px;
                 border-radius:4px; margin:0; }}
  /* print: fit the brief on one A4 page */
  @page {{ size:A4; margin:10mm; }}
  @media print {{
    body {{ background:#fff; font-size:10.5px; line-height:1.35; }}
    main {{ margin:0; border:0; padding:0; max-width:none; }}
    h1 {{ font-size:18px; }} h2 {{ margin:9px 0 4px; font-size:10px; }}
    th, td {{ padding:2px 6px; font-size:9.5px; }}
    img {{ max-height:78mm; width:auto; margin:2px auto; }}
    .call {{ padding:5px 10px; }} .call strong {{ font-size:16px; }}
    .disclaimer {{ font-size:8.5px; padding:5px 8px; }}
    p {{ margin:4px 0; }}
  }}
  @media (max-width:600px) {{ main {{ padding:18px 16px; margin:0; border-radius:0; }} }}
</style></head>
<body><main>
{body}
</main></body></html>"""


def render_html(md_text: str, title: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables"])
    return HTML_TEMPLATE.format(title=html.escape(title), body=body)


def write_report(summary, df, scored, sentiment, signal, news, out_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Write <ticker>_brief.md, <ticker>_brief.html (chart embedded) and <ticker>_chart.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker = summary["ticker"]
    png = render_chart(df, ticker)
    paths = {"chart": out_dir / f"{ticker}_chart.png", "markdown": out_dir / f"{ticker}_brief.md",
             "html": out_dir / f"{ticker}_brief.html"}
    paths["chart"].write_bytes(png)
    paths["markdown"].write_text(build_markdown(summary, df, scored, sentiment, signal, news, paths["chart"].name))
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode()
    md_inline = build_markdown(summary, df, scored, sentiment, signal, news, data_uri)
    paths["html"].write_text(render_html(md_inline, f"{ticker} equity research brief"))
    return paths


if __name__ == "__main__":
    import sys

    from data_pipeline import DEFAULT_TICKER, run_pipeline
    from llm_reasoning import aggregate_sentiment, generate_signal, get_client, score_headlines, setup_logging

    setup_logging()
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    try:
        client = get_client()
    except RuntimeError as exc:
        sys.exit(f"Config error: {exc}")
    df, news, summary = run_pipeline(ticker)
    if "error" in summary:
        sys.exit(f"No price data for {ticker}; nothing to report.")
    scored, failed = score_headlines(news, ticker, summary.get("company"), client)
    sentiment = aggregate_sentiment(scored, len(failed))
    signal = generate_signal(df, ticker, summary.get("company"), sentiment, client)
    for kind, path in write_report(summary, df, scored, sentiment, signal, news).items():
        print(f"{kind:>8}: {path}")
