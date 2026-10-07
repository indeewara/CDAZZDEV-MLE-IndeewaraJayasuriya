"""Task 3A - a single tool-using research agent built as an explicit LangGraph loop:

    START -> agent --(tool call?)--> tools -> agent -> ... -> (no tool call) -> END

The LLM decides every step from the observations so far (no fixed tool order). A tool round budget, per-model
token pacing, and error-returning tools keep it from stalling or crashing. A MemorySaver checkpointer keeps the
conversation per thread_id, which is the short-term memory used for follow-up questions (Task 3C).
"""
import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent_prompts import (GROUNDING_REPAIR, REPAIR_USER, REPORT_SYSTEM, REPORT_USER, RESEARCH_AGENT_SYSTEM,
                           RESEARCH_QUERY)
from agent_schemas import ResearchReport, ungrounded_numbers
from tracing import (MODEL_FALLBACKS, call_json, current_agent, ensure_groq_key, estimate_tokens, groq_client,
                     EXHAUSTED_MODELS, is_daily_quota_error, limiter, log_event)
import tracing
from tools import TOOLS

AGENT_MODEL = "openai/gpt-oss-120b"   # strongest free Groq model with tool calling
REASONING_EFFORT = "low"              # enough to plan the next tool call; keeps tokens inside the free-tier budget
MAX_TOOL_ROUNDS = 8                   # 5 tools + room for retries/alternatives; stops runaway loops
MAX_OBSERVATION_CHARS = 2500          # cap per tool result in the conversation (token budget)
THOUGHT_PREVIEW_CHARS = 300
# Market conventions a report may state without a tool having returned them (documented, deliberately short)
KNOWN_FACTS = "Conventions: 1 option contract = 100 shares; 252 trading days per year; 90-day horizon."
GROUNDING_RETRIES = 2              # re-ask the report writer at most twice to drop numbers not in the observations


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_rounds: int


def _observation(result) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_OBSERVATION_CHARS else text[:MAX_OBSERVATION_CHARS] + "...[truncated]"


def _show(name: str, msg: AIMessage) -> None:
    if not tracing.VERBOSE:
        return
    thought = (msg.additional_kwargs.get("reasoning_content") or "").strip().replace("\n", " ")
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
    fallback = fallback_with_tools = None
    if llm is None:
        ensure_groq_key()
        llm = ChatGroq(model=AGENT_MODEL, temperature=0, reasoning_effort=REASONING_EFFORT, max_retries=2)
        fallback = ChatGroq(model=MODEL_FALLBACKS[AGENT_MODEL], temperature=0, reasoning_effort=REASONING_EFFORT,
                            max_retries=2)
        fallback_with_tools = fallback.bind_tools(tools, parallel_tool_calls=False)
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)  # one call per step -> visible observe/decide cycles
    already_out = fallback is not None and AGENT_MODEL in EXHAUSTED_MODELS   # quota ran out earlier this session
    active = {"model": MODEL_FALLBACKS[AGENT_MODEL] if already_out else AGENT_MODEL, "fallen_back": already_out}
    allowed = {t.name: t for t in tools}
    system = SystemMessage(system_prompt.format(max_rounds=max_tool_rounds))

    def agent_node(state: AgentState):
        current_agent.set(name)
        messages = [system, *state["messages"]]
        exhausted = state.get("tool_rounds", 0) >= max_tool_rounds
        if exhausted:
            messages.append(HumanMessage("Tool budget used up. Reply now with your findings from the observations."))
        try:
            reply = _invoke(exhausted, messages)
        except Exception as exc:  # noqa: BLE001 - an LLM outage must not crash the run
            if is_daily_quota_error(exc) and fallback is not None and not active["fallen_back"]:
                active.update(model=MODEL_FALLBACKS[AGENT_MODEL], fallen_back=True)
                EXHAUSTED_MODELS.add(AGENT_MODEL)
                log_event({"event": "model_fallback", "from": AGENT_MODEL, "to": active["model"], "reason": "daily quota"})
                if tracing.VERBOSE:
                    print(f"[{name}] {AGENT_MODEL} daily quota used up - falling back to {active['model']}")
                try:
                    reply = _invoke(exhausted, messages)
                except Exception as exc2:  # noqa: BLE001
                    reply = _llm_unavailable(exc2)
            else:
                reply = _llm_unavailable(exc)
        _show(name, reply)
        return {"messages": [reply]}

    def _invoke(exhausted: bool, messages: list) -> AIMessage:
        if active["fallen_back"]:
            model = fallback if exhausted else fallback_with_tools
        else:
            model = llm if exhausted else llm_with_tools
        lim = limiter(active["model"])
        lim.wait(estimate_tokens("".join(str(m.content) for m in messages)))
        reply = model.invoke(messages)
        lim.record((reply.usage_metadata or {}).get("total_tokens", estimate_tokens(str(messages))))
        return reply

    def _llm_unavailable(exc: Exception) -> AIMessage:
        log_event({"event": "llm_unavailable", "error": f"{type(exc).__name__}: {exc}"[:200]})
        if tracing.VERBOSE:
            print(f"[{name}] language model unavailable ({type(exc).__name__}) - stopping with the findings so far")
        return AIMessage(content="The language model is unavailable (quota or outage); stopping with the observations "
                                 "gathered so far.")

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
