# Task 3 - Multi-Agent Financial Research System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indeewara/CDAZZDEV-MLE-IndeewaraJayasuriya/blob/main/task3_agentic/task3_multi_agent_research.ipynb)
&nbsp; Notebook: [`task3_multi_agent_research.ipynb`](task3_multi_agent_research.ipynb) &nbsp;·&nbsp; Trace:
[`logs/agent_trace.jsonl`](logs/agent_trace.jsonl)

Agents that answer *"Analyse the current financial health and market sentiment of [TICKER]. Identify the top three
risks to its share price over the next 90 days and suggest one data-driven hedge strategy."* - deciding which tools to
call from what they observe, handing structured data to each other, remembering context, and logging every tool call.

**Stack:** LangGraph (explicit agent ⇄ tools graph), `openai/gpt-oss-120b` via the Groq free tier (falls back to
`gpt-oss-20b` if the daily quota runs out), Pydantic for every structured output, yfinance, DuckDuckGo search (`ddgs`).

## 3A - Tool-using research agent
| Tool | Returns |
|---|---|
| `get_price_data(ticker, period)` | OHLCV + SMA 50/200, RSI, MACD, Bollinger, 52-week range, P/E, momentum (reuses Task 1's tested pipeline) |
| `get_news(ticker, n)` | recent headlines (title, publisher, date) |
| `calculate_volatility(ticker, window)` | annualised volatility vs its 1-year range, expected 90-day 1σ move **and the resulting price band** |
| `llm_sentiment(headlines)` | per-headline labels + confidence-weighted aggregate (one batched LLM call, validated with Task 1's schema) |
| `web_search(query)` | analyst commentary, price targets, events |

- **Autonomous:** the graph is `agent → (tool call?) → tools → agent`; the model chooses every call, one per step
  (`parallel_tool_calls=False`), so each observe → decide cycle is visible. No tool order exists in the code.
- **Errors never stop the run:** tools return `{"error": ...}` instead of raising; bad arguments, refused tools and LLM
  outages become observations. Demonstrated by forcing `get_news` to fail - the agent switches to `web_search`.
- **Report:** a validated `ResearchReport` with exactly three sections (financial health, three risks with
  evidence, hedge). **Every number in it is checked against the tool results**; invented figures are sent back for
  removal. Hedge strikes come from the volatility tool's price band, so the LLM does no option arithmetic.

## 3B - Two agents, restricted tools, critique loop
| Agent | Tools (enforced in the graph) | Output |
|---|---|---|
| A - Data Analyst | get_price_data, calculate_volatility, llm_sentiment | `DataBrief` |
| B - Research Writer | web_search, get_news | `MultiAgentReport` |

`DataBrief` (A→B) → B researches → `ClarificationRequest` (B→A) → A answers with its tools → `ClarificationResponse`
(A→B) → B's final report. Neither agent can produce news sentiment alone (B has headlines, A has the sentiment tool),
so in practice B sends its headlines and A scores them. All three hand-offs are Pydantic objects, printed in full and
logged; their numbers are grounding-checked. B's report must use A's figures **in its analysis** - this is checked,
not just claimed.

## 3C - Memory and observability
- **Short-term:** LangGraph `MemorySaver` keeps each thread's messages; `ask_followup()` asks in the same thread and
  counts tool calls from the trace - the notebook's follow-up is answered with 0.
- **Persistent:** the final brief is saved to `cache/<TICKER>_<YYYY-MM-DD>.json`; `research_with_cache()` loads it on
  a same-day request instead of running any tool (next day = new key = fresh research).
- **Trace:** [`logs/agent_trace.jsonl`](logs/agent_trace.jsonl) - one line per event; each tool call has tool,
  inputs, output (truncated to 200 characters), duration, status, agent and session. Hand-offs, grounding checks,
  cache hits and model fallbacks are logged too.

## Files
| File | Purpose |
|---|---|
| [`tools.py`](tools.py) | the five tools |
| [`agent.py`](agent.py) | 3A agent graph, report writer, number-grounding retries, model fallback |
| [`multi_agent.py`](multi_agent.py) | 3B pipeline |
| [`memory.py`](memory.py) | 3C cache and follow-up memory |
| [`tracing.py`](tracing.py) | trace logging, token pacing for the free tier, validated JSON calls |
| [`agent_prompts.py`](agent_prompts.py), [`agent_schemas.py`](agent_schemas.py) | all prompts; Pydantic schemas and grounding checks |
| [`test_agent.py`](test_agent.py) | 22 offline tests - a scripted model drives the real graph |

## Run
```bash
source .venv/bin/activate                        # GROQ_API_KEY in .env
python task3_agentic/agent.py AAPL               # 3A
python task3_agentic/multi_agent.py MSFT         # 3B
pytest task3_agentic                             # offline tests
```
A full notebook run uses ~130k tokens of `gpt-oss-120b`'s 200k free daily quota (8k tokens/minute; calls are paced).

## Limitations
- Figures are grounding-checked, reasoning is not: a real number can still be misread (e.g. a price called "close
  to" a band it is far from). An LLM-as-judge pass, as in Task 2, would be the next step.
- News is not cross-checked against data: a headline can contradict the prices and still be cited.
- With the fallback model the reports are noticeably weaker; the run completes, but quality drops.
- Not investment advice.
