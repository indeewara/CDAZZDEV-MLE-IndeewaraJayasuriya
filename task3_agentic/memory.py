"""Task 3C - memory.

Short-term memory: the agent graph is compiled with a LangGraph MemorySaver checkpointer, so every message (including
tool results) is kept per thread_id. ask_followup() sends a new question into the same thread; the agent can answer
from the earlier observations. The number of tool calls made while answering is counted from the trace - 0 means the
answer came from memory.

Persistent memory: the final report is saved to cache/<TICKER>_<YYYY-MM-DD>.json. research_with_cache() checks for
today's file first and loads it instead of re-running any tool. Keying by date means a cached brief is never more
than a day old - the next day's run researches afresh.
"""
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import tracing
from agent_schemas import MultiAgentReport, ResearchReport
from tracing import log_event

CACHE_DIR = Path(__file__).resolve().parent / "cache"
REPORT_TYPES = {"ResearchReport": ResearchReport, "MultiAgentReport": MultiAgentReport}


# ======================================================================================
# Persistent cache
# ======================================================================================
def cache_path(ticker: str, day: date | None = None) -> Path:
    return CACHE_DIR / f"{ticker.strip().upper()}_{(day or date.today()).isoformat()}.json"


def save_report(report: ResearchReport, session_id: str, day: date | None = None) -> Path:
    path = cache_path(report.ticker, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ticker": report.ticker.upper(), "date": (day or date.today()).isoformat(),
                                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "session_id": session_id, "report_type": type(report).__name__,
                                "report": report.model_dump()}, indent=2, ensure_ascii=False))
    log_event({"event": "cache_saved", "path": path.name})
    return path


def load_report(ticker: str, day: date | None = None) -> ResearchReport | None:
    """Today's cached report, or None. A corrupt or unreadable file is ignored (logged), never fatal."""
    path = cache_path(ticker, day)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return REPORT_TYPES[data["report_type"]].model_validate(data["report"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        log_event({"event": "cache_invalid", "path": path.name, "error": f"{type(exc).__name__}: {exc}"[:200]})
        return None


def _tool_calls_logged(session: str) -> int:
    if not tracing.TRACE_PATH.exists():
        return 0
    return sum(1 for line in tracing.TRACE_PATH.read_text().splitlines()
               if (r := json.loads(line)).get("event") == "tool_call" and r.get("session_id") == session)


def research_with_cache(ticker: str, runner) -> tuple[ResearchReport | None, dict]:
    """Load today's cached report for `ticker` if present; otherwise run `runner(ticker)` (3A run_research or the
    3B run_pipeline) and cache its report. Returns (report, info) with source = 'cache' or 'fresh run'."""
    ticker = ticker.strip().upper()
    start = time.perf_counter()
    cached = load_report(ticker)
    if cached is not None:
        sid = tracing.new_session()
        log_event({"event": "cache_hit", "ticker": ticker, "path": cache_path(ticker).name})
        info = {"source": "cache", "path": str(cache_path(ticker)), "session_id": sid,
                "seconds": round(time.perf_counter() - start, 3), "tool_calls": _tool_calls_logged(sid)}
        if tracing.VERBOSE:
            print(f"cache HIT: loaded {cache_path(ticker).name} - no tools run "
                  f"({info['tool_calls']} tool calls, {info['seconds']} s)")
        return cached, info

    log_event({"event": "cache_miss", "ticker": ticker})
    if tracing.VERBOSE:
        print(f"cache MISS: no {cache_path(ticker).name} - running the agents")
    report, run = runner(ticker)
    info = {"source": "fresh run", "session_id": run.get("session_id"), "run": run,
            "seconds": round(time.perf_counter() - start, 1),
            "tool_calls": _tool_calls_logged(run.get("session_id", ""))}
    if report is not None:
        info["path"] = str(save_report(report, run.get("session_id", "")))
        if tracing.VERBOSE:
            print(f"saved {Path(info['path']).name} ({info['tool_calls']} tool calls, {info['seconds']} s)")
    return report, info


# ======================================================================================
# Short-term memory
# ======================================================================================
def ask_followup(graph, config: dict, question: str) -> dict:
    """Ask a follow-up in the same thread. Returns the answer and how many tools were called to produce it,
    counted two ways: tool calls in the new messages, and tool_call events in the trace for this session."""
    session = tracing.session_id.get()
    before_state = graph.get_state(config).values
    n_before = len(before_state.get("messages", []))
    logged_before = _tool_calls_logged(session)
    log_event({"event": "followup_question", "question": question})

    state = graph.invoke({"messages": [HumanMessage(question)]}, config)
    new = state["messages"][n_before:]
    calls = [c["name"] for m in new if isinstance(m, AIMessage) for c in m.tool_calls]
    answer = next((m.content for m in reversed(new) if isinstance(m, AIMessage) and m.content), "")
    result = {"answer": answer, "tool_calls_in_messages": calls,
              "tool_calls_in_trace": _tool_calls_logged(session) - logged_before,
              "messages_in_memory_before": n_before}
    log_event({"event": "followup_answer", "tool_calls": len(calls), "answer": answer[:200]})
    return result
