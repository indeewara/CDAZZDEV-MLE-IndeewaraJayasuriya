# Reflection

## Architectural decisions

**Across all tasks:** code computes facts; the LLM only reasons over them. Every LLM output is validated with
Pydantic, and failures are sent back once for repair before being logged and handled. Prompts live in separate
modules, and scripted fake models let the logic be tested offline (80 tests).

**Task 1:** indicators are computed from first principles and checked against Wilder's RSI example and independent
loop implementations. The signal prompt receives derived relationships (distance from moving averages,
days since crosses, values now versus ten days ago) rather than raw values, plus one explicit rule (trend sets the
base case, momentum moves the call towards Hold), which made calls consistent across stocks.

**Task 2:** I chose structured news-impact analysis over plain sentiment: a custom 11-type taxonomy and JSON format give
fine-tuning something measurable to learn. Diversity was planned in code (balanced specs across event type, sector,
company, direction, magnitude and style) instead of left to the teacher. A blind second labeller filtered items whose
text did not support their label (65% yield), and a grounding check rejected invented names and numbers. QLoRA
trained Qwen2.5-1.5B with loss on assistant tokens only; the 600-token system prompt would otherwise dominate.
Validation loss fell every epoch (0.945, 0.881, 0.862).

On the 18 held-out items, valid JSON rose from 0% to 100%, magnitude accuracy from 28% to 61% and ROUGE-L from
0.19 to 0.24; the judge's "incorrect" verdicts fell from 22% to 0%.

**Task 3:** an explicit LangGraph loop with one tool call per step makes every observe-decide cycle visible. Tools
return errors instead of raising, so outages become observations: with `get_news` forced to fail, the agent switched
to web search unprompted. The volatility tool computes the hedge price bands so the LLM does no arithmetic, and every
number in a report or hand-off is checked against tool results. The tools are split so that neither agent can produce
news sentiment alone, which gives the critique loop real work; the code checks that the answer is used in the
analysis, not just claimed.

## Limitations encountered

- **Free-tier limits shaped the design.** Groq allows 8k tokens per minute and 200k per day; development hit the daily
  cap, which led to pacing, compact tool outputs and a logged fallback to `gpt-oss-20b`, whose reports were weaker.
  Llama-3-70B, suggested in the brief, has been retired on Groq.
- **Task 2 labels come from the teacher**, so evaluation measures agreement with a strong model, not market truth. The
  judge shares a family with the student, which if anything favours the base model.
- **Data gaps:** routine-versus-significant insider sales stayed ambiguous (only 12 examples), dates cluster in
  October, and the set is 47% US.
- **Grounding checks numbers, not reasoning:** one report called a price "close to" a band 12% away. News is also not
  cross-checked against data; a "bear market" headline was cited while the price sat near its 52-week high.
- **Fragile tooling:** transformers 5 removed `warmup_ratio`, bf16 adapters broke fp16 training on the T4, and
  Colab's torchao blocked the merge. Output varied between runs even at temperature 0.

## With more time

- A human-labelled test set of 50+ items for Task 2, so improvement is measured against ground truth.
- An LLM-as-judge pass on Task 3 reports to check reasoning, and a step that cross-checks news against prices.
- Use the fine-tuned Task 2 model as the event-analysis tool in Tasks 1 and 3, replacing headline-only sentiment.
- Backtest the hedge recommendations and the Buy/Hold/Sell signal on historical data.
