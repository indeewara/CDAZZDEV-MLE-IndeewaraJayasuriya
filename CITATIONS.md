# Citations

This file documents all AI assistance, external models, data sources and published methods used in this repository,
in the format required by Section 2.2 of the assessment brief.

## 1. AI coding assistant

All code, tests, notebooks and documentation in this repository were written with **Claude Code** (Anthropic), using
the models **Claude Opus 5.5** (`claude-opus-5-5`) and **Claude Sonnet 5.5** (`claude-sonnet-5-5`), in an interactive
session on 2026-10-06 and 2026-10-07. I directed the work step by step: I chose the scope of each step, reviewed
outputs, ran the code and the Colab notebooks, reported errors back, and asked for each part to be checked against the
assessment rubric. The design decisions are discussed in `REFLECTION.md`.

The prompts below summarise the requests made in that interactive session (several were agreed in steps), each
followed by the files it produced.

### Task 1 - `task1_financial/`
```text
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Propose a repository structure for the three tasks', Date: 2026-10-06
  Files: folder layout, .gitignore, .env.example
# AI-ASSISTED: Claude (claude-sonnet-5-5), Prompt: 'Fetch at least two years of daily OHLCV data with yfinance using a relative period, with no hardcoded dates', Date: 2026-10-06
  Files: data_pipeline.py (fetch_ohlcv)
# AI-ASSISTED: Claude (claude-sonnet-5-5), Prompt: 'Compute SMA 50/200, RSI 14 with Wilder smoothing, MACD (12, 26, 9) and Bollinger Bands (20, 2) from first principles without TA-Lib, with tests against known values', Date: 2026-10-06
  Files: data_pipeline.py (rsi_wilder, add_indicators), test_indicators.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Retrieve at least ten recent headlines from a free source with fallbacks, and build a summary dictionary with price, 52-week range, P/E, YTD return and a momentum signal', Date: 2026-10-06
  Files: data_pipeline.py (fetch_news, build_summary, momentum_signal)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Test the pipeline against missing and malformed data and fix every case that raises an unhandled exception', Date: 2026-10-06
  Files: data_pipeline.py (robustness fixes), test_indicators.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add independent cross-checks for MACD and Bollinger Bands, diversify the headline sample, add a network timeout and remove magic numbers', Date: 2026-10-06
  Files: data_pipeline.py, test_indicators.py
# AI-ASSISTED: Claude (claude-sonnet-5-5), Prompt: 'Score each headline with an LLM as validated JSON (headline, sentiment, confidence, brief_reason) and aggregate into an overall sentiment score', Date: 2026-10-06
  Files: llm_reasoning.py (score_headlines, aggregate_sentiment), schemas.py, prompts.py, test_llm_reasoning.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate a Buy/Hold/Sell signal with a 3-5 sentence justification that reasons over combinations of indicators rather than restating values', Date: 2026-10-06
  Files: llm_reasoning.py (build_technical_context, generate_signal), prompts.py, schemas.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Validate every LLM response with Pydantic, log and repair failures, and keep all prompts separate from business logic', Date: 2026-10-06
  Files: llm_reasoning.py (call_llm_json), test_llm_reasoning.py
# AI-ASSISTED: Claude (claude-sonnet-5-5), Prompt: 'Render a one-page equity research brief from the 1A and 1B outputs as Markdown and styled HTML with an embedded matplotlib chart and a risk disclaimer', Date: 2026-10-06
  Files: report.py, outputs/AAPL_brief.md, outputs/AAPL_brief.html, outputs/AAPL_chart.png
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build an executed Task 1 notebook that demonstrates every requirement with visible outputs', Date: 2026-10-06
  Files: task1_equity_research.ipynb, README.md
```

### Task 2 - `task2_genai/`
```text
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Compare candidate domain-specific use cases and write a structured problem statement with input, output and correctness criteria', Date: 2026-10-06
  Files: README.md (problem statement)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build a synthetic data pipeline with a teacher model: code-planned diversity, blind second-model labelling, schema and grounding validation, deduplication and a stratified 80/10/10 split', Date: 2026-10-06
  Files: generate_data.py, news_prompts.py, news_schemas.py, test_generate_data.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Report dataset diversity, validate the chat-format JSONL against the base model template, and publish the full teacher prompts', Date: 2026-10-06
  Files: dataset_report.py, TEACHER_PROMPTS.md, outputs/dataset_diversity.png
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement QLoRA fine-tuning (4-bit NF4) of Qwen2.5-1.5B-Instruct on a Colab T4, set and justify every hyperparameter explicitly, and merge the adapter', Date: 2026-10-06
  Files: finetune.py, task2_news_impact_finetuning.ipynb
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Evaluate base versus fine-tuned models on the test set with ROUGE-L, BERTScore, field accuracy, an LLM-as-judge and a manual hallucination review', Date: 2026-10-06
  Files: evaluate.py, test_evaluate.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Diagnose and fix Colab errors during training and merging (fp16/bf16 gradient scaling, stale imports, incompatible torchao, model card)', Date: 2026-10-06
  Files: finetune.py, task2_news_impact_finetuning.ipynb
```
The choice of use case (structured financial news impact analysis instead of plain sentiment) was also discussed in a
separate AI chat session, whose suggestions I brought into this one and reviewed before deciding.

### Task 3 - `task3_agentic/`
```text
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement five research tools and a LangGraph agent that chooses tools from its observations, recovers from tool failures and writes a three-section report with grounded numbers', Date: 2026-10-07
  Files: tools.py, agent.py, tracing.py, agent_prompts.py, agent_schemas.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build a two-agent pipeline (Data Analyst and Research Writer) with enforced tool access, Pydantic hand-offs and a critique loop', Date: 2026-10-07
  Files: multi_agent.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add short-term memory for follow-up questions, a persistent JSON cache keyed by ticker and date, and a model fallback for quota limits', Date: 2026-10-07
  Files: memory.py, tracing.py, agent.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Check the multi-agent pipeline against the rubric and verify that the clarification is incorporated into the analysis', Date: 2026-10-07
  Files: multi_agent.py, agent_schemas.py (incorporates_clarification)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 3 notebook, README and offline tests that drive the agent graph with a scripted model', Date: 2026-10-07
  Files: task3_multi_agent_research.ipynb, README.md, test_agent.py
```

### Repository-level
```text
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Draft CITATIONS.md in the required format and a REFLECTION.md under 600 words covering all tasks', Date: 2026-10-07
  Files: CITATIONS.md, REFLECTION.md
```

## 2. LLMs used inside the system (Groq free tier)

| Model | Role | Prompts |
|---|---|---|
| `openai/gpt-oss-120b` | Task 1 headline sentiment and Buy/Hold/Sell signal; **Task 2 teacher (data generation)**; Task 3 agents and report writer | `task1_financial/prompts.py`, `task2_genai/news_prompts.py`, `task3_agentic/agent_prompts.py` |
| `openai/gpt-oss-20b` | Task 2 blind labeller (quality filter); Task 3 batched sentiment tool and quota fallback | same files |
| `qwen/qwen3.8-27b` | Task 2 LLM-as-judge | `task2_genai/news_prompts.py` (`JUDGE_SYSTEM`) |

**Teacher-model data generation:** the full system prompt and user template, plus the blind labeller's and the
training prompts, are reproduced verbatim in [`task2_genai/TEACHER_PROMPTS.md`](task2_genai/TEACHER_PROMPTS.md)
(generated from `news_prompts.py`) and printed in the Task 2 notebook.

**Fine-tuned model:** `Qwen/Qwen2.5-1.5B-Instruct` (Alibaba Qwen team, Apache-2.0), fine-tuned with QLoRA.
Smoke tests used `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`.

## 3. Methods and published sources (no code copied)

No open-source code was copied or adapted; libraries were used through their public APIs. Formulas and settings
follow these sources:

- J. Welles Wilder Jr., *New Concepts in Technical Trading Systems* (1978) - RSI with Wilder smoothing. Reference
  values in `test_indicators.py` are from the StockCharts ChartSchool RSI worked example
  (https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi).
- Gerald Appel - MACD (12, 26, 9); John Bollinger - Bollinger Bands (20, 2σ, population standard deviation).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023), arXiv:2305.14314 - NF4, double
  quantization, paged 8-bit AdamW, learning rate 2e-4, max grad norm 0.3, LoRA on all linear layers.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), arXiv:2106.09685.
- Lin, *ROUGE* (2004); Zhang et al., *BERTScore* (2020), arXiv:1904.09675.
- Library documentation: Hugging Face Transformers, TRL (`SFTTrainer`, prompt-completion format), PEFT, bitsandbytes,
  LangGraph, LangChain-Groq, yfinance, feedparser, ddgs, rouge-score, bert-score.
- Chart colours: a colour-blind-validated categorical palette from Claude Code's data-visualisation guidance.

## 4. Data sources

- Prices, P/E, company info: Yahoo Finance via `yfinance`.
- News headlines: Google News RSS and Yahoo Finance RSS (public feeds; the yfinance news endpoint returned no items
  during development).
- Analyst commentary (Task 3): DuckDuckGo search via `ddgs`.
- Task 2 training data: synthetic - real company names with fictional events written by the teacher model.
