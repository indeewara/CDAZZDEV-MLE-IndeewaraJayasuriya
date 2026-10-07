"""Observability and shared LLM plumbing for Task 3.

- traced(): wraps every tool. Each call is appended to logs/agent_trace.jsonl with tool name, inputs, output
  (truncated to 200 characters), wall-clock duration, status and the calling agent. Exceptions never escape a
  tool: they become {"error": ...} results the agent can react to.
- TokenRateLimiter: keeps each Groq model under its free-tier tokens-per-minute limit.
- call_json(): one validated JSON call with a repair retry (same pattern as Task 1).
"""
import contextvars
import functools
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACE_PATH = HERE / "logs" / "agent_trace.jsonl"
MAX_OUTPUT_CHARS = 200          # the assessment's truncation length for logged outputs
GROQ_TOKENS_PER_MINUTE = 8000   # free-tier TPM for gpt-oss-120b and gpt-oss-20b (from Groq response headers)
# Each Groq model has its own free daily token quota (200k for gpt-oss-120b). When the main model's quota is used up,
# switch to the smaller model rather than fail - a graceful-degradation fallback, logged to the trace.
MODEL_FALLBACKS = {"openai/gpt-oss-120b": "openai/gpt-oss-20b"}


EXHAUSTED_MODELS: set[str] = set()   # models whose daily quota ran out this session - skip straight to the fallback


def active_model(model: str) -> str:
    return MODEL_FALLBACKS.get(model, model) if model in EXHAUSTED_MODELS else model


def is_daily_quota_error(exc: Exception) -> bool:
    return "per day" in str(exc).lower() or "tokens per day" in str(exc).lower()

current_agent = contextvars.ContextVar("current_agent", default="research_agent")
session_id = contextvars.ContextVar("session_id", default="none")
FAILURE_INJECTION: set[str] = set()   # tool names forced to fail - used only to demonstrate error handling
VERBOSE = True


def new_session() -> str:
    sid = uuid.uuid4().hex[:8]
    session_id.set(sid)
    return sid


def log_event(record: dict) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "session_id": session_id.get(),
              "agent": current_agent.get(), **record}
    with TRACE_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def traced(fn):
    """Log every call of a tool function; turn exceptions into error results instead of crashing the agent."""
    @functools.wraps(fn)
    def wrapper(**kwargs):
        start = time.perf_counter()
        if fn.__name__ in FAILURE_INJECTION:
            result = {"error": f"{fn.__name__} unavailable (simulated outage for the error-handling demo)"}
        else:
            try:
                result = fn(**kwargs)
            except Exception as exc:  # noqa: BLE001 - any tool failure becomes data for the agent
                result = {"error": f"{type(exc).__name__}: {exc}"}
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        text = json.dumps(result, ensure_ascii=False, default=str)
        status = "error" if isinstance(result, dict) and "error" in result else "ok"
        log_event({"event": "tool_call", "tool": fn.__name__, "inputs": kwargs, "output": text[:MAX_OUTPUT_CHARS],
                   "output_chars": len(text), "duration_ms": duration_ms, "status": status})
        if VERBOSE:
            print(f"    [{current_agent.get()}] tool {fn.__name__}({_short(kwargs)}) -> {status} in {duration_ms:.0f} ms: "
                  f"{text[:150]}{'...' if len(text) > 150 else ''}")
        return result
    return wrapper


def _short(kwargs: dict) -> str:
    return ", ".join(f"{k}={v!r}"[:80] for k, v in kwargs.items())


class TokenRateLimiter:
    """Sliding 60-second window: wait before a call if it would push the minute's usage over the limit."""
    def __init__(self, tokens_per_minute: int = GROQ_TOKENS_PER_MINUTE):
        self.limit, self.window = tokens_per_minute, deque()

    def _used(self) -> int:
        while self.window and time.time() - self.window[0][0] > 60:
            self.window.popleft()
        return sum(t for _, t in self.window)

    def wait(self, estimated_tokens: int) -> None:
        estimated_tokens = min(estimated_tokens, self.limit)  # a single call larger than the limit just waits for an empty window
        while self._used() + estimated_tokens > self.limit:
            time.sleep(1)

    def record(self, tokens: int) -> None:
        self.window.append((time.time(), tokens))


LIMITERS: dict[str, TokenRateLimiter] = {}


def limiter(model: str) -> TokenRateLimiter:
    """One limiter per model: Groq's TPM limits are per model."""
    return LIMITERS.setdefault(model, TokenRateLimiter())


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 200   # conservative chars-per-token plus room for the reply


def ensure_groq_key() -> None:
    """Key from the environment (Colab secret) or the repo-root .env - never from code."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set - add it to .env (see .env.example)")


def groq_client():
    from groq import Groq
    ensure_groq_key()
    return Groq(max_retries=5)


def call_json(client, model: str, system: str, user: str, schema: type[BaseModel], repair_template: str,
              max_attempts: int = 3, reasoning_effort: str = "low") -> BaseModel | None:
    """JSON-mode call validated against `schema`; validation errors are sent back for a corrected reply."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    model = active_model(model)
    for attempt in range(1, max_attempts + 1):
        lim = limiter(model)
        lim.wait(estimate_tokens(system + user))
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0,
                                                  reasoning_effort=reasoning_effort,
                                                  response_format={"type": "json_object"})
        except Exception as exc:  # noqa: BLE001 - API errors: retry, then give up gracefully
            if is_daily_quota_error(exc) and model in MODEL_FALLBACKS:
                EXHAUSTED_MODELS.add(model)
                log_event({"event": "model_fallback", "from": model, "to": MODEL_FALLBACKS[model], "reason": "daily quota"})
                print(f"    {model} daily quota used up - falling back to {MODEL_FALLBACKS[model]}")
                model = MODEL_FALLBACKS[model]
                continue
            print(f"    LLM error (attempt {attempt}/{max_attempts}): {type(exc).__name__}: {str(exc)[:150]}")
            continue
        lim.record(resp.usage.total_tokens)
        raw = (resp.choices[0].message.content or "") if resp.choices else ""
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            print(f"    validation failed (attempt {attempt}/{max_attempts}) for {schema.__name__}: {exc.errors()[0]['msg']}")
            log_event({"event": "validation_error", "schema": schema.__name__, "attempt": attempt,
                       "error": str(exc)[:300]})
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": repair_template.format(errors=exc)}]
    return None
