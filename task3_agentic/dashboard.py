"""Bonus - Streamlit dashboard for logs/agent_trace.jsonl.

Run: streamlit run task3_agentic/dashboard.py
Pick a session to see its tool-call timeline per agent, hand-offs, latency per tool and every logged event.
"""
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

TRACE_PATH = Path(__file__).resolve().parent / "logs" / "agent_trace.jsonl"
# first three slots of a colour-blind-validated categorical palette, one per agent; others share a neutral grey
AGENT_COLOURS = {"research_agent": "#2a78d6", "data_analyst": "#eb6834", "research_writer": "#1baf7a"}
OTHER_COLOUR = "#8a8984"


def load_events(path: Path = TRACE_PATH) -> pd.DataFrame:
    """All trace events as a DataFrame (malformed lines are skipped, not fatal)."""
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def sessions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session: when it started, what it was about, how many tool calls it made."""
    if df.empty:
        return pd.DataFrame(columns=["session_id", "started", "label", "tool_calls"])
    out = []
    for sid, g in df.groupby("session_id"):
        start, hit = g[g["event"] == "pipeline_start"], g[g["event"] == "cache_hit"]
        tickers = [i.get("ticker") for i in g.get("inputs", pd.Series(dtype=object)).dropna() if isinstance(i, dict)]
        if not start.empty:
            label = f"3B pipeline - {start.iloc[0].get('ticker', '')}"
        elif not hit.empty:
            label = f"cache hit - {hit.iloc[0].get('ticker', '')}"
        elif tickers:
            label = f"agent run - {tickers[0]}"
        else:
            label = "other events"
        out.append({"session_id": sid, "started": g["ts"].min(), "label": label,
                    "tool_calls": int((g["event"] == "tool_call").sum())})
    return pd.DataFrame(out).sort_values("started", ascending=False)


def summary(g: pd.DataFrame) -> dict:
    calls = g[g["event"] == "tool_call"]
    return {"tool calls": len(calls),
            "errors": int((calls.get("status") == "error").sum()) if not calls.empty else 0,
            "tool time (s)": round(float(calls["duration_ms"].sum()) / 1000, 1) if not calls.empty else 0.0,
            "hand-offs": int((g["event"] == "handoff").sum()),
            "model fallbacks": int((g["event"] == "model_fallback").sum())}


def timeline(calls: pd.DataFrame) -> pd.DataFrame:
    """Start/end of each tool call (the logged timestamp is when it finished), in seconds since the session's first
    call, one numbered row per call in the order the agents made them."""
    t = calls.sort_values("ts").copy()
    t["end"] = t["ts"]
    t["start"] = t["end"] - t["duration_ms"].apply(lambda ms: timedelta(milliseconds=float(ms)))
    t0 = t["start"].min()
    t["start_s"] = (t["start"] - t0).dt.total_seconds()
    t["end_s"] = (t["end"] - t0).dt.total_seconds()
    t["row"] = [f"{i}. {tool}" + (" (error)" if st == "error" else "")
                for i, (tool, st) in enumerate(zip(t["tool"], t["status"]), 1)]
    return t


def main() -> None:
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Agent trace", layout="wide")
    st.title("Agent trace dashboard")
    st.caption(f"Reads `{TRACE_PATH.relative_to(TRACE_PATH.parents[2])}` - every tool call with inputs, output "
               "(first 200 characters), duration and status; plus hand-offs, grounding checks, cache and fallback events.")

    df = load_events()
    if df.empty:
        st.warning("No trace yet - run the agents first.")
        return
    sess = sessions(df)
    options = {f"{r.started:%Y-%m-%d %H:%M} · {r.label} · {r.tool_calls} tool calls · {r.session_id}": r.session_id
               for r in sess.itertuples()}
    keys = list(options)
    wanted = st.query_params.get("session")          # ?session=<id> opens a specific run
    index = next((i for i, k in enumerate(keys) if options[k] == wanted), 0)
    chosen = options[st.sidebar.selectbox("Session", keys, index=index)]
    g = df[df["session_id"] == chosen].sort_values("ts")
    agents = sorted(g["agent"].dropna().unique())
    shown = st.sidebar.multiselect("Agents", agents, default=agents)
    g = g[g["agent"].isin(shown)]

    for col, (name, value) in zip(st.columns(5), summary(g).items()):
        col.metric(name, value)

    calls = g[g["event"] == "tool_call"]
    if not calls.empty:
        st.subheader("Tool-call timeline")
        t = timeline(calls)
        domain = list(t["agent"].unique())
        colours = [AGENT_COLOURS.get(a, OTHER_COLOUR) for a in domain]
        chart = alt.Chart(t).mark_bar(cornerRadius=3).encode(
            x=alt.X("start_s:Q", title="seconds since the first tool call"), x2="end_s:Q",
            y=alt.Y("row:N", sort=list(t["row"]), title=None, axis=alt.Axis(labelOverlap=False, labelLimit=260)),
            color=alt.Color("agent:N", scale=alt.Scale(domain=domain, range=colours), legend=alt.Legend(orient="top")),
            tooltip=["agent", "tool", "status", "duration_ms", alt.Tooltip("output:N", title="output (200 chars)")])
        # one row per call: size the chart per row (Altair "step"), so rows never collapse or hide labels
        st.altair_chart(chart.properties(height=alt.Step(32)), width="stretch")

        left, right = st.columns(2)
        left.subheader("Latency per tool")
        left.bar_chart(calls.groupby("tool")["duration_ms"].mean().rename("mean ms"), horizontal=True)
        right.subheader("Status")
        right.bar_chart(calls["status"].value_counts().rename("calls"), horizontal=True)

    handoffs = g[g["event"] == "handoff"]
    if not handoffs.empty:
        st.subheader("Hand-offs between agents")
        for h in handoffs.to_dict("records"):        # records, not itertuples: "from" is a Python keyword
            with st.expander(f"{h['ts']:%H:%M:%S} · {h['from']} → {h['to']} · {h['schema']}"):
                st.json(h["payload"])

    st.subheader("All events")
    cols = [c for c in ["ts", "agent", "event", "tool", "inputs", "output", "duration_ms", "status", "ungrounded_numbers"]
            if c in g.columns]
    st.dataframe(g[cols].astype(str), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
