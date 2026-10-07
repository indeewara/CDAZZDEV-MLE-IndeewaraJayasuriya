"""Task 3B - two-agent pipeline with restricted tools, Pydantic hand-offs and a critique loop.

  Agent A (Data Analyst)    tools: get_price_data, calculate_volatility, llm_sentiment
  Agent B (Research Writer) tools: web_search, get_news

  1. A researches the numbers                     -> DataBrief            (A -> B)
  2. B gathers news and analyst commentary
  3. B sends one specific request                 -> ClarificationRequest (B -> A)
  4. A answers with its tools / earlier results   -> ClarificationResponse (A -> B)
  5. B writes the final report incorporating it   -> MultiAgentReport

Neither agent can produce news sentiment alone: B has the headlines but no sentiment tool; A has the sentiment tool
but no way to fetch headlines. Tool access is enforced in the graph (calls to other tools are refused), not just
stated in the prompt. Every hand-off is a validated Pydantic object, printed and logged to agent_trace.jsonl, and
its numbers are checked against the tool observations they came from.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import tracing
from agent import AGENT_MODEL, build_agent, grounded_json, observations
from agent_prompts import (ANALYST_FOLLOWUP, ANALYST_TASK, CLARIFICATION_RESPONSE_SYSTEM, CLARIFICATION_RESPONSE_USER,
                           CLARIFICATION_SYSTEM, CLARIFICATION_USER, DATA_ANALYST_SYSTEM, DATA_BRIEF_SYSTEM,
                           DATA_BRIEF_USER, INCORPORATION_REPAIR, MULTI_REPORT_SYSTEM, MULTI_REPORT_USER, REPAIR_USER,
                           RESEARCH_QUERY, RESEARCH_WRITER_SYSTEM, WRITER_TASK)
from agent_schemas import (ClarificationRequest, ClarificationResponse, DataBrief, MultiAgentReport,
                           incorporates_clarification)
from tools import TOOLS
from tracing import call_json, current_agent, groq_client, log_event

DATA_ANALYST, RESEARCH_WRITER = "data_analyst", "research_writer"
DATA_ANALYST_TOOLS = ["get_price_data", "calculate_volatility", "llm_sentiment"]
RESEARCH_WRITER_TOOLS = ["web_search", "get_news"]
AGENT_TOOL_ROUNDS = 5          # each agent has a narrower job than the 3A agent, so a smaller budget
EXTRACT_MODEL = "openai/gpt-oss-20b"   # structuring a hand-off from observations is simple; uses a separate quota


def _banner(title: str) -> None:
    if tracing.VERBOSE:
        print(f"\n{'=' * 18} {title} {'=' * 18}")


def handoff(sender: str, receiver: str, payload) -> None:
    """Log and show a structured message between agents."""
    log_event({"event": "handoff", "from": sender, "to": receiver, "schema": type(payload).__name__,
               "payload": payload.model_dump()})
    if tracing.VERBOSE:
        print(f"\n>>> HAND-OFF {sender} -> {receiver} [{type(payload).__name__}]\n"
              f"{json.dumps(payload.model_dump(), indent=2, ensure_ascii=False)}")


def build_team(memory=None):
    memory = memory or MemorySaver()
    analyst = build_agent([TOOLS[n] for n in DATA_ANALYST_TOOLS], DATA_ANALYST_SYSTEM, name=DATA_ANALYST,
                          max_tool_rounds=AGENT_TOOL_ROUNDS, checkpointer=memory)
    writer = build_agent([TOOLS[n] for n in RESEARCH_WRITER_TOOLS], RESEARCH_WRITER_SYSTEM, name=RESEARCH_WRITER,
                         max_tool_rounds=AGENT_TOOL_ROUNDS, checkpointer=memory)
    return analyst, writer


def _final_message(messages: list) -> str:
    return next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content), "")


def run_pipeline(ticker: str) -> tuple[MultiAgentReport | None, dict]:
    ticker = ticker.strip().upper()
    sid = tracing.new_session()
    question = RESEARCH_QUERY.format(ticker=ticker)
    analyst, writer = build_team()
    a_cfg = {"configurable": {"thread_id": f"{sid}-analyst"}, "recursion_limit": 30}  # A keeps its own memory
    b_cfg = {"configurable": {"thread_id": f"{sid}-writer"}, "recursion_limit": 30}
    log_event({"event": "pipeline_start", "ticker": ticker, "question": question})
    print(f"session {sid} | {question}")

    # 1. Agent A: quantitative research -> DataBrief
    _banner("1. Agent A - Data Analyst researches the numbers")
    a_state = analyst.invoke({"messages": [HumanMessage(ANALYST_TASK.format(ticker=ticker))], "tool_rounds": 0}, a_cfg)
    a_obs = observations(a_state["messages"])
    # the brief is B's only window onto the numbers, so it is written by the stronger model (gpt-oss-20b lost detail)
    brief, _ = grounded_json(DATA_BRIEF_SYSTEM, DATA_BRIEF_USER.format(ticker=ticker, observations=a_obs), DataBrief,
                             a_obs, model=AGENT_MODEL, label=DATA_ANALYST)
    if brief is None:
        return None, {"session_id": sid, "failed_at": "data_brief"}
    handoff(DATA_ANALYST, RESEARCH_WRITER, brief)
    brief_json = brief.model_dump_json(indent=2)

    # 2. Agent B: qualitative research, shaped by the brief
    _banner("2. Agent B - Research Writer gathers news and commentary")
    b_state = writer.invoke({"messages": [HumanMessage(WRITER_TASK.format(question=question, brief=brief_json))],
                             "tool_rounds": 0}, b_cfg)
    b_obs = observations(b_state["messages"])

    # 3. Critique loop, request: B -> A
    _banner("3. Critique loop - Agent B asks Agent A for one specific piece of data")
    current_agent.set(RESEARCH_WRITER)
    request = call_json(groq_client(), AGENT_MODEL, CLARIFICATION_SYSTEM,
                        CLARIFICATION_USER.format(brief=brief_json, observations=b_obs), ClarificationRequest, REPAIR_USER)
    if request is None:
        return None, {"session_id": sid, "failed_at": "clarification_request"}
    handoff(RESEARCH_WRITER, DATA_ANALYST, request)

    # 4. Critique loop, response: A answers in the same thread (it remembers its earlier work)
    _banner("4. Critique loop - Agent A answers")
    before = len(a_state["messages"])
    a_state = analyst.invoke({"messages": [HumanMessage(ANALYST_FOLLOWUP.format(
        question=request.question, reason=request.reason, headlines=json.dumps(request.headlines) or "none"))],
        "tool_rounds": 0}, a_cfg)
    followup_obs = observations(a_state["messages"][before:])
    response, _ = grounded_json(
        CLARIFICATION_RESPONSE_SYSTEM,
        CLARIFICATION_RESPONSE_USER.format(question=request.question, observations=followup_obs or "(answered from earlier results)",
                                           summary=_final_message(a_state["messages"])),
        ClarificationResponse, followup_obs + " " + a_obs, model=EXTRACT_MODEL, label=DATA_ANALYST)
    if response is None:
        return None, {"session_id": sid, "failed_at": "clarification_response"}
    handoff(DATA_ANALYST, RESEARCH_WRITER, response)

    # 5. Agent B writes the final report, incorporating the answer
    _banner("5. Agent B writes the final report")
    sources = " ".join([a_obs, b_obs, followup_obs, question])
    user = MULTI_REPORT_USER.format(question=question, brief=brief_json, observations=b_obs,
                                    request=request.model_dump_json(), response=response.model_dump_json())
    report, ungrounded = grounded_json(MULTI_REPORT_SYSTEM, user, MultiAgentReport, sources, label=RESEARCH_WRITER)
    # "incorporate", checked rather than claimed: A's figures must appear in the analysis, not only in the
    # section where B says it used them. One retry with an explicit instruction if they do not.
    incorporated = report is not None and incorporates_clarification(report, response)
    if report is not None and not incorporated:
        figures = ", ".join(f"{k}={v}" for k, v in response.key_figures.items()) or response.answer[:150]
        print(f"[{RESEARCH_WRITER}] the analysis does not use the Data Analyst's answer yet - revising")
        report, ungrounded = grounded_json(MULTI_REPORT_SYSTEM, user + INCORPORATION_REPAIR.format(figures=figures),
                                           MultiAgentReport, sources, label=RESEARCH_WRITER)
        incorporated = report is not None and incorporates_clarification(report, response)
    log_event({"event": "critique_incorporated", "incorporated": incorporated})
    if tracing.VERBOSE:
        print(f"[{RESEARCH_WRITER}] Data Analyst's answer used in the report's analysis: {'YES' if incorporated else 'NO'}")
    log_event({"event": "pipeline_end", "ticker": ticker, "report_ok": report is not None, "ungrounded": ungrounded,
               "clarification_incorporated": incorporated})
    return report, {"session_id": sid, "brief": brief, "request": request, "response": response,
                    "ungrounded": ungrounded, "incorporated": incorporated, "analyst": analyst, "writer": writer, "a_cfg": a_cfg, "b_cfg": b_cfg}


if __name__ == "__main__":
    import sys
    report, run = run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
    print("\n" + (report.to_markdown() if report else f"PIPELINE FAILED at {run.get('failed_at')}"))
