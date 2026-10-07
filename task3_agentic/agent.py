"""Task 3A - a single tool-using research agent built as an explicit LangGraph loop:

    START -> agent --(tool call?)--> tools -> agent -> ... -> (no tool call) -> END

The LLM decides every step from the observations so far (no fixed tool order). A tool round budget, per-model
token pacing, and error-returning tools keep it from stalling or crashing. A MemorySaver checkpointer keeps the
conversation per thread_id, which is the short-term memory used for follow-up questions (Task 3C).
"""
import json
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent_prompts import (GROUNDING_REPAIR, REPAIR_USER, REPORT_SYSTEM, REPORT_USER, RESEARCH_AGENT_SYSTEM,
                           RESEARCH_QUERY)
from agent_schemas import ResearchReport, ungrounded_numbers
from tracing import (EXHAUSTED_MODELS, MAX_OUTPUT_TOKENS, MODEL_ORDER, PER_MINUTE_RETRY_WAIT_S, active_model, call_json,
                     current_agent, ensure_groq_key, estimate_tokens, groq_client, is_daily_quota_error,
                     is_per_minute_limit, is_tool_format_error, limiter, log_event, next_model, reasoning_effort)
import tracing
from tools import TOOLS

AGENT_MODEL = "openai/gpt-oss-120b"   # strongest free Groq model with tool calling; falls back along MODEL_ORDER
REASONING_EFFORT = "low"              # gpt-oss: enough to plan the next tool call within the free-tier budget
MAX_TOOL_ROUNDS = 8                   # 5 tools + room for retries/alternatives; stops runaway loops
MAX_OBSERVATION_CHARS = 2500          # cap per tool result in the conversation (token budget)
THOUGHT_PREVIEW_CHARS = 300
# Market conventions a report may state without a tool having returned them (documented, deliberately short)
KNOWN_FACTS = "Conventions: 1 option contract = 100 shares; 252 trading days per year; 90-day horizon."
MAX_FORMAT_RETRIES = 2              # malformed tool-call JSON: retry once, then let the next model take this step
MAX_RATE_LIMIT_WAITS = 3            # per step: wait out a per-minute limit at most 3 times before giving up
GROUNDING_RETRIES = 2              # re-ask the report writer at most twice to drop numbers not in the observations


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_rounds: int


def _observation(result) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_OBSERVATION_CHARS else text[:MAX_OBSERVATION_CHARS] + "...[truncated]"


def rationale(msg: AIMessage) -> str:
    """Why the agent took this step: the sentence it wrote before the tool call (the prompts ask for one) or,
    for models that return it separately (gpt-oss), its reasoning text."""
    text = (msg.content if msg.tool_calls and msg.content else "") or msg.additional_kwargs.get("reasoning_content") or ""
    return str(text).strip().replace("\n", " ")


def _show(name: str, msg: AIMessage) -> None:
    if not tracing.VERBOSE:
        return
    thought = rationale(msg)
    if thought:
        print(f"[{name}] thinks: {thought[:THOUGHT_PREVIEW_CHARS]}{'...' if len(thought) > THOUGHT_PREVIEW_CHARS else ''}")
    for call in msg.tool_calls:
        print(f"[{name}] decides: call {call['name']}({json.dumps(call['args'])[:160]})")
    if not msg.tool_calls and msg.content:
        print(f"[{name}] answers: {str(msg.content)[:600]}")


def build_agent(tools: list, system_prompt: str, name: str = "research_agent", max_tool_rounds: int = MAX_TOOL_ROUNDS,
                checkpointer=None, llm=None):
    """Compile an agent graph restricted to `tools`. Calls to any other tool name are refused (3B tool access).
    `llm` is injectable so the loop can be tested offline with a scripted model."""
    # One client per model in the fallback chain; an injected llm (tests) is used as the only "model".
    if llm is None:
        ensure_groq_key()
        clients = {m: ChatGroq(model=m, temperature=0, reasoning_effort=reasoning_effort(m, REASONING_EFFORT),
                               max_tokens=MAX_OUTPUT_TOKENS, max_retries=2) for m in MODEL_ORDER}
    else:
        clients = {"injected": llm}
    with_tools = {m: c.bind_tools(tools, parallel_tool_calls=False)  # one call per step -> visible observe/decide
                  for m, c in clients.items()}
    allowed = {t.name: t for t in tools}
    system = SystemMessage(system_prompt.format(max_rounds=max_tool_rounds))

    def current_model() -> str:
        return "injected" if "injected" in clients else active_model(AGENT_MODEL)

    def agent_node(state: AgentState):
        current_agent.set(name)
        messages = [system, *state["messages"]]
        exhausted = state.get("tool_rounds", 0) >= max_tool_rounds
        if exhausted:
            messages.append(HumanMessage("Tool budget used up. Reply now with your findings from the observations."))
        waits, format_errors, step_model = 0, 0, None
        while True:   # daily quota -> next model; per-minute limit -> wait; malformed tool call -> retry; else end
            model = step_model or current_model()
            try:
                reply = _invoke(model, exhausted, messages)
                break
            except Exception as exc:  # noqa: BLE001 - an LLM outage must not crash the run
                if is_per_minute_limit(exc) and waits < MAX_RATE_LIMIT_WAITS:
                    waits += 1
                    log_event({"event": "rate_limit_wait", "model": model, "seconds": PER_MINUTE_RETRY_WAIT_S})
                    if tracing.VERBOSE:
                        print(f"[{name}] {model} per-minute limit - waiting {PER_MINUTE_RETRY_WAIT_S} s")
                    time.sleep(PER_MINUTE_RETRY_WAIT_S)
                    continue
                if is_tool_format_error(exc) and format_errors < MAX_FORMAT_RETRIES and model != "injected":
                    # retry once with the same model, then hand THIS step to the next model (quota not marked used)
                    format_errors += 1
                    step_model = model if format_errors == 1 else (next_model(model) or model)
                    log_event({"event": "tool_call_format_retry", "model": model, "retry_with": step_model})
                    if tracing.VERBOSE:
                        print(f"[{name}] {model} produced a malformed tool call - retrying with {step_model}")
                    continue
                nxt = None
                if is_daily_quota_error(exc) and model != "injected":
                    EXHAUSTED_MODELS.add(model)
                    nxt = next_model(model)
                if nxt is None:
                    reply = _llm_unavailable(exc)
                    break
                log_event({"event": "model_fallback", "from": model, "to": nxt, "reason": "daily quota"})
                if tracing.VERBOSE:
                    print(f"[{name}] {model} daily quota used up - falling back to {nxt}")
        _show(name, reply)
        return {"messages": [reply]}

    def _invoke(model: str, exhausted: bool, messages: list) -> AIMessage:
        runnable = clients[model] if exhausted else with_tools[model]
        lim = limiter(model)
        lim.wait(estimate_tokens("".join(str(m.content) for m in messages)))
        reply = runnable.invoke(messages)
        lim.record((reply.usage_metadata or {}).get("total_tokens", estimate_tokens(str(messages))))
        return reply

    def _llm_unavailable(exc: Exception) -> AIMessage:
        log_event({"event": "llm_unavailable", "error": f"{type(exc).__name__}: {exc}"[:200]})
        if tracing.VERBOSE:
            print(f"[{name}] language model unavailable ({type(exc).__name__}) - stopping with the findings so far")
        return AIMessage(content=f"The language model call failed ({type(exc).__name__}); stopping with the "
                                 "observations gathered so far.")

    def tool_node(state: AgentState):
        current_agent.set(name)
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = allowed.get(call["name"])
            if tool is None:   # enforced restriction, not just a prompt instruction
                result = {"error": f"tool {call['name']!r} is not available to {name}; available: {sorted(allowed)}"}
                log_event({"event": "tool_refused", "tool": call["name"], "inputs": call["args"]})
            else:
                try:
                    result = tool.invoke(call["args"])   # traced: logged to agent_trace.jsonl
                except Exception as exc:  # noqa: BLE001 - e.g. wrong argument types from the LLM
                    result = {"error": f"invalid arguments for {call['name']}: {exc}"}
            results.append(ToolMessage(content=_observation(result), tool_call_id=call["id"], name=call["name"]))
        return {"messages": results, "tool_rounds": state.get("tool_rounds", 0) + 1}

    def route(state: AgentState):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def observations(messages: list) -> str:
    """Tool results in call order, as compact text for the report writer."""
    calls = {c["id"]: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    lines = []
    for m in messages:
        if isinstance(m, ToolMessage):
            c = calls.get(m.tool_call_id, {})
            lines.append(f"- {m.name}({json.dumps(c.get('args', {}))}): {m.content}")
    return "\n".join(lines)


def grounded_json(system: str, user: str, schema, sources: str, model: str = AGENT_MODEL,
                  label: str = "report_writer"):
    """Validated JSON (schema + repair) whose numbers must all appear in `sources`; numbers that appear nowhere
    are sent back for removal, up to GROUNDING_RETRIES times. Returns (object or None, numbers still ungrounded).
    Used for the 3A report and for every 3B hand-off, so no agent passes on invented figures."""
    current_agent.set(label)
    client, obj, bad = groq_client(), None, []
    for attempt in range(GROUNDING_RETRIES + 1):
        extra = GROUNDING_REPAIR.format(numbers=", ".join(bad)) if bad else ""
        obj = call_json(client, model, system, user + extra, schema, REPAIR_USER)
        if obj is None:
            return None, []
        bad = ungrounded_numbers(obj.model_dump_json(), sources + " " + KNOWN_FACTS)
        log_event({"event": "grounding_check", "schema": schema.__name__, "attempt": attempt + 1,
                   "ungrounded_numbers": bad})
        if tracing.VERBOSE:
            print(f"[{label}] grounding check {attempt + 1} ({schema.__name__}): "
                  + (f"numbers not in observations: {bad}" if bad else "every number traced to a tool result"))
        if not bad:
            break
    if bad and tracing.VERBOSE:
        print(f"[{label}] WARNING: after {GROUNDING_RETRIES + 1} attempts these numbers are still not traceable to a "
              f"tool result: {bad} - treat them as unverified")
    return obj, bad


def write_report(messages: list, question: str) -> tuple[ResearchReport | None, list[str]]:
    """Turn the agent's observations into the validated, grounded three-section report."""
    final = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content), "")
    obs = observations(messages)
    user = REPORT_USER.format(question=question, observations=obs, summary=final)
    return grounded_json(REPORT_SYSTEM, user, ResearchReport, obs + " " + question)


def run_research(ticker: str, graph=None, thread_id: str | None = None) -> tuple[ResearchReport | None, dict]:
    """3A end to end: the agent researches autonomously, then the report is written from its observations."""
    sid = tracing.new_session()
    graph = graph or build_agent(list(TOOLS.values()), RESEARCH_AGENT_SYSTEM, checkpointer=MemorySaver())
    question = RESEARCH_QUERY.format(ticker=ticker.upper())
    config = {"configurable": {"thread_id": thread_id or sid}, "recursion_limit": 4 * MAX_TOOL_ROUNDS}
    print(f"session {sid} | question: {question}\n")
    state = graph.invoke({"messages": [HumanMessage(question)], "tool_rounds": 0}, config)
    report, ungrounded = write_report(state["messages"], question)
    return report, {"graph": graph, "config": config, "state": state, "session_id": sid, "ungrounded": ungrounded}


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    report, run = run_research(ticker)
    print("\n" + (report.to_markdown() if report else "REPORT FAILED - see logs/agent_trace.jsonl"))
