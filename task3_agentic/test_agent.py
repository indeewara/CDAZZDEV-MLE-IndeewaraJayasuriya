"""Offline tests for Task 3A (no network, no API calls): the agent loop driven by a scripted model,
tool restriction, error handling, the tool budget, trace logging, report schema and grounding."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

import tracing
from agent import build_agent, observations
from agent_schemas import ResearchReport, ungrounded_numbers
from tools import TOOLS


@pytest.fixture(autouse=True)
def tmp_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(tracing, "VERBOSE", False)
    return tmp_path / "trace.jsonl"


class ScriptedLLM:
    """Stands in for ChatGroq: returns queued AIMessages and records what it was shown."""
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def bind_tools(self, tools, **_):
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


def call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])


def _echo(x: int) -> dict:
    """Echo a number."""
    return {"echo": x}


def _boom(x: int) -> dict:
    """Always fails."""
    raise RuntimeError("upstream down")


ECHO = StructuredTool.from_function(tracing.traced(_echo), name="_echo", description="Echo a number.")
BOOM = StructuredTool.from_function(tracing.traced(_boom), name="_boom", description="Always fails.")


def run(llm, tools, max_rounds=8):
    graph = build_agent(tools, "system {max_rounds}", name="test_agent", max_tool_rounds=max_rounds, llm=llm)
    return graph.invoke({"messages": [HumanMessage("go")], "tool_rounds": 0}, {"recursion_limit": 40})


def test_observe_then_decide_cycle():
    llm = ScriptedLLM([call("_echo", {"x": 7}), AIMessage(content="done: 7")])
    state = run(llm, [ECHO])
    kinds = [type(m).__name__ for m in state["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    second_call_context = llm.seen[1]
    assert any(isinstance(m, ToolMessage) and '"echo": 7' in m.content for m in second_call_context)  # it observed


def test_tool_outside_allowed_set_is_refused(tmp_trace):
    state = run(ScriptedLLM([call("web_search", {"query": "x"}), AIMessage(content="ok")]), [ECHO])
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "not available" in tool_msg.content
    assert json.loads(tmp_trace.read_text().splitlines()[0])["event"] == "tool_refused"


def test_tool_exception_becomes_observation_and_loop_continues():
    state = run(ScriptedLLM([call("_boom", {"x": 1}), call("_echo", {"x": 2}, "c2"), AIMessage(content="recovered")]),
                [ECHO, BOOM])
    tool_msgs = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    assert "upstream down" in tool_msgs[0] and '"echo": 2' in tool_msgs[1]
    assert state["messages"][-1].content == "recovered"


def test_bad_arguments_become_observation():
    state = run(ScriptedLLM([call("_echo", {"x": "not-a-number"}), AIMessage(content="ok")]), [ECHO])
    assert "invalid arguments" in next(m for m in state["messages"] if isinstance(m, ToolMessage)).content


def test_tool_budget_forces_an_answer():
    llm = ScriptedLLM([call("_echo", {"x": 1}), AIMessage(content="final")])
    state = run(llm, [ECHO], max_rounds=1)
    assert state["messages"][-1].content == "final"
    assert "Tool budget used up" in llm.seen[1][-1].content


def test_trace_record_has_required_fields(tmp_trace):
    TOOLS["calculate_volatility"].invoke({"ticker": "AAPL", "window": 2})   # validation error path, no network
    rec = json.loads(tmp_trace.read_text().splitlines()[-1])
    assert {"tool", "inputs", "output", "duration_ms", "status", "ts", "agent", "session_id"} <= set(rec)
    assert rec["status"] == "error" and len(rec["output"]) <= tracing.MAX_OUTPUT_CHARS


def test_failure_injection_and_long_output_truncated(tmp_trace, monkeypatch):
    monkeypatch.setattr(tracing, "FAILURE_INJECTION", {"get_news"})
    assert "simulated outage" in TOOLS["get_news"].invoke({"ticker": "AAPL"})["error"]
    big = StructuredTool.from_function(tracing.traced(lambda: {"blob": "x" * 5000}), name="big", description="big")
    big.invoke({})
    assert len(json.loads(tmp_trace.read_text().splitlines()[-1])["output"]) == tracing.MAX_OUTPUT_CHARS


def test_tool_input_errors_without_network():
    assert "unsupported period" in TOOLS["get_price_data"].invoke({"ticker": "AAPL", "period": "7y"})["error"]
    assert "no headlines" in TOOLS["llm_sentiment"].invoke({"headlines": []})["error"]


def test_observations_pairs_calls_with_results():
    msgs = [call("_echo", {"x": 3}), ToolMessage(content='{"echo": 3}', tool_call_id="c1", name="_echo")]
    assert observations(msgs) == '- _echo({"x": 3}): {"echo": 3}'


REPORT = {"ticker": "AAPL", "financial_health_summary": "Apple trades at $333.63 near its 52-week high with a P/E of 38.",
          "top_risks": [{"title": f"Risk {i}", "evidence": "RSI fell from 66.4 to 53.8 in ten days.", "severity": "High"}
                        for i in range(3)],
          "hedge_strategy": {"strategy": "Protective put", "instrument": "AAPL 90-day puts", "sizing": "strike $291.67",
                             "rationale": "Strike at the 1-sigma 90-day downside band from calculate_volatility."},
          "data_sources": ["get_price_data", "calculate_volatility"]}


def test_report_requires_exactly_three_risks():
    assert ResearchReport.model_validate(REPORT).top_risks[0].severity == "high"
    with pytest.raises(ValidationError):
        ResearchReport.model_validate({**REPORT, "top_risks": REPORT["top_risks"][:2]})
    assert "## Top Three Risks" in ResearchReport.model_validate(REPORT).to_markdown()


def test_grounding_flags_invented_numbers_only():
    src = "current_price 333.63 pe_ratio 38.1755 price_90d_minus_1sigma 291.67 rsi 66.4 53.8 shares 191,753"
    assert ungrounded_numbers("$333.6, P/E 38, put $291.67, RSI 66.4 -> 53.8, 191,753 shares, 3 risks, Q3", src) == []
    assert ungrounded_numbers("App Store ~15% of income, a 4% drop, P/E 22x, put at $300", src) == ["4", "15", "22", "300"]


# ---- 3B: tool restriction, hand-off schemas, hand-off logging ------------------------------------
from agent_schemas import ClarificationRequest, DataBrief, MultiAgentReport  # noqa: E402
from multi_agent import DATA_ANALYST_TOOLS, RESEARCH_WRITER_TOOLS, handoff  # noqa: E402


def _restricted_run(tool_names, forbidden_call):
    graph = build_agent([TOOLS[n] for n in tool_names], "sys {max_rounds}", name="x",
                        llm=ScriptedLLM([call(forbidden_call, {"ticker": "AAPL"}), AIMessage(content="ok")]))
    state = graph.invoke({"messages": [HumanMessage("go")], "tool_rounds": 0}, {"recursion_limit": 20})
    return next(m for m in state["messages"] if isinstance(m, ToolMessage)).content


def test_tool_access_is_enforced_per_agent():
    assert set(DATA_ANALYST_TOOLS) & set(RESEARCH_WRITER_TOOLS) == set()
    assert "web_search" not in DATA_ANALYST_TOOLS and "get_price_data" not in RESEARCH_WRITER_TOOLS
    assert "not available" in _restricted_run(RESEARCH_WRITER_TOOLS, "get_price_data")     # writer can't fetch prices
    assert "not available" in _restricted_run(DATA_ANALYST_TOOLS, "get_news")               # analyst can't fetch news


BRIEF = {"ticker": "MSFT", "current_price": 529.3, "high_52w": 549.2, "low_52w": 348.5, "sma_50": 493.1,
         "sma_200": 432.0, "rsi_14": 68.0, "pe_ratio": 29.3, "ytd_return_pct": 12.0, "annualised_volatility_pct": 22.08,
         "one_year_avg_volatility_pct": 30.06, "volatility_percentile_1y": 31.0, "expected_90d_move_pct": 13.19,
         "price_90d_minus_1sigma": 459.46, "price_90d_plus_1sigma": 599.14, "max_drawdown_1y_pct": -34.5,
         "key_findings": ["Price 529.3 sits above both SMA-50 and SMA-200, a confirmed uptrend",
                          "Volatility 22.08% is below its one-year average of 30.06%",
                          "RSI 68 is close to the overbought level of 70"]}


def test_data_brief_rejects_findings_that_only_echo_fields():
    DataBrief.model_validate(BRIEF)
    with pytest.raises(ValidationError):
        DataBrief.model_validate({**BRIEF, "key_findings": ["current_price 529.3", "pe 29.3", "vol 22.08"]})


def test_clarification_request_limits_headlines():
    ClarificationRequest(question="Score these headlines for me", reason="The brief has no sentiment.", headlines=["a"])
    with pytest.raises(ValidationError):
        ClarificationRequest(question="Score these headlines for me", reason="Too many items.", headlines=["h"] * 16)


def test_handoff_is_logged_with_payload(tmp_trace):
    req = ClarificationRequest(question="Score these headlines for me", reason="The brief has no sentiment.")
    handoff("research_writer", "data_analyst", req)
    rec = json.loads(tmp_trace.read_text().splitlines()[-1])
    assert rec["event"] == "handoff" and rec["from"] == "research_writer" and rec["schema"] == "ClarificationRequest"
    assert rec["payload"]["question"] == "Score these headlines for me"


def test_multi_agent_report_shows_how_clarification_was_used():
    md = MultiAgentReport.model_validate({**REPORT, "clarification_incorporated": "Asked A to score 10 headlines; +0.57."}).to_markdown()
    assert "## How the Data Analyst's clarification was used" in md and "## Hedge Strategy Recommendation" in md


# ---- 3C: persistent cache and short-term memory ---------------------------------------------------
import memory as mem  # noqa: E402


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(mem, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def test_cache_round_trip_keyed_by_ticker_and_date(tmp_cache):
    report = ResearchReport.model_validate(REPORT)
    path = mem.save_report(report, "s1")
    assert path.name.startswith("AAPL_") and path.name.endswith(".json") and path.parent == tmp_cache
    assert mem.load_report("aapl") == report
    from datetime import date, timedelta
    assert mem.load_report("AAPL", date.today() - timedelta(days=1)) is None   # yesterday's key: not found


def test_second_run_loads_cache_without_running_tools(tmp_cache):
    calls = []

    def runner(ticker):
        calls.append(ticker)
        return ResearchReport.model_validate(REPORT), {"session_id": "s-fresh"}

    first, info1 = mem.research_with_cache("AAPL", runner)
    second, info2 = mem.research_with_cache("AAPL", runner)
    assert calls == ["AAPL"]                                   # the agents ran exactly once
    assert info1["source"] == "fresh run" and info2["source"] == "cache"
    assert second == first and info2["tool_calls"] == 0


def test_corrupt_cache_is_ignored_not_fatal(tmp_cache):
    tmp_cache.mkdir(parents=True)
    mem.cache_path("AAPL").write_text("{not json")
    assert mem.load_report("AAPL") is None


def test_followup_answered_from_memory_without_tools():
    from langgraph.checkpoint.memory import MemorySaver
    llm = ScriptedLLM([call("_echo", {"x": 21}), AIMessage(content="found 21"),
                       AIMessage(content="Earlier you asked; the value was 21.")])
    graph = build_agent([ECHO], "sys {max_rounds}", name="mem", llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}
    graph.invoke({"messages": [HumanMessage("get x")], "tool_rounds": 0}, cfg)
    result = mem.ask_followup(graph, cfg, "What was x?")
    assert result["tool_calls_in_messages"] == [] and result["tool_calls_in_trace"] == 0
    assert "21" in result["answer"] and result["messages_in_memory_before"] == 4
    assert any(isinstance(m, ToolMessage) and "21" in m.content for m in llm.seen[-1])  # the old result was in context


def test_llm_outage_ends_run_gracefully_instead_of_crashing(tmp_trace):
    class Down(ScriptedLLM):
        def invoke(self, messages):
            raise RuntimeError("Rate limit reached ... tokens per day (TPD)")
    state = run(Down([]), [ECHO])
    assert "unavailable" in state["messages"][-1].content
    assert any(json.loads(l)["event"] == "llm_unavailable" for l in tmp_trace.read_text().splitlines())


def test_incorporation_is_checked_in_the_analysis_not_just_claimed():
    from agent_schemas import ClarificationResponse, incorporates_clarification
    resp = ClarificationResponse(answer="Ten headlines scored 0.5677 overall, positive.", key_figures={"score": 0.5677})
    claimed_only = MultiAgentReport.model_validate({**REPORT, "clarification_incorporated": "I used the 0.5677 score."})
    assert not incorporates_clarification(claimed_only, resp)          # only mentioned in its own section
    used = {**REPORT, "financial_health_summary": REPORT["financial_health_summary"] + " Headline sentiment is +0.5677.",
            "clarification_incorporated": "Asked A to score headlines; used its 0.5677 in the summary."}
    assert incorporates_clarification(MultiAgentReport.model_validate(used), resp)
    text_only = ClarificationResponse(answer="Momentum is still bullish across both windows.")
    assert incorporates_clarification(claimed_only, text_only)         # nothing numeric to trace


# ---- Bonus: trace dashboard (data functions only; no Streamlit server) ---------------------------
def test_dashboard_reads_trace_and_labels_sessions(tmp_path):
    import dashboard as d
    path = tmp_path / "t.jsonl"
    rows = [{"ts": "2026-10-07T01:00:00+00:00", "session_id": "p1", "agent": "data_analyst", "event": "pipeline_start", "ticker": "MSFT"},
            {"ts": "2026-10-07T01:00:02+00:00", "session_id": "p1", "agent": "data_analyst", "event": "tool_call", "tool": "get_price_data",
             "inputs": {"ticker": "MSFT"}, "output": "{}", "duration_ms": 1500.0, "status": "ok"},
            {"ts": "2026-10-07T01:00:05+00:00", "session_id": "p1", "agent": "research_writer", "event": "handoff", "from": "a", "to": "b"},
            {"ts": "2026-10-07T02:00:00+00:00", "session_id": "c1", "agent": "research_agent", "event": "cache_hit", "ticker": "AAPL"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    df = d.load_events(path)
    labels = dict(zip(d.sessions(df)["session_id"], d.sessions(df)["label"]))
    assert labels == {"p1": "3B pipeline - MSFT", "c1": "cache hit - AAPL"}
    assert d.summary(df[df.session_id == "p1"]) == {"tool calls": 1, "errors": 0, "tool time (s)": 1.5, "hand-offs": 1, "model fallbacks": 0}
    t = d.timeline(df[df.event == "tool_call"])
    assert (t["end_s"] - t["start_s"]).iloc[0] == 1.5 and t["start_s"].iloc[0] == 0 and t["row"].iloc[0] == "1. get_price_data"
