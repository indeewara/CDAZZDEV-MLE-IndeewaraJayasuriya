# CDAZZDEV Senior ML Engineer Assessment - Indeewara Jayasuriya

All three tasks: **Financial AI**, **Generative AI** and **Agentic Workflows**. Everything runs on free tiers (Groq,
Google Colab T4, Hugging Face Hub); no API key is stored anywhere in the repository.

| Task | Notebook | What it delivers |
|---|---|---|
| **1 - Financial AI** · [README](task1_financial/README.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indeewara/CDAZZDEV-MLE-IndeewaraJayasuriya/blob/main/task1_financial/task1_equity_research.ipynb) | yfinance pipeline with five indicators from first principles, news, summary dict; LLM headline sentiment and a Buy/Hold/Sell signal reasoned over indicator relationships, all Pydantic-validated; **bonus:** one-page HTML research brief with an embedded chart |
| **2 - Generative AI** · [README](task2_genai/README.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indeewara/CDAZZDEV-MLE-IndeewaraJayasuriya/blob/main/task2_genai/task2_news_impact_finetuning.ipynb) | Financial news impact analysis: 185-example synthetic dataset (teacher + blind second labeller), QLoRA fine-tune of Qwen2.5-1.5B with every hyperparameter justified, base-vs-fine-tuned evaluation. **Model:** [Indee99/qwen2.5-1.5b-financial-news-impact](https://huggingface.co/Indee99/qwen2.5-1.5b-financial-news-impact) |
| **3 - Agentic Workflows** · [README](task3_agentic/README.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indeewara/CDAZZDEV-MLE-IndeewaraJayasuriya/blob/main/task3_agentic/task3_multi_agent_research.ipynb) | LangGraph research agent with five tools; two-agent pipeline with enforced tool access, Pydantic hand-offs and a critique loop; short-term and persistent memory; [`agent_trace.jsonl`](task3_agentic/logs/agent_trace.jsonl); **bonus:** Streamlit trace dashboard |

**Also in the root:** [`CITATIONS.md`](CITATIONS.md) (AI assistance, models, sources) and
[`REFLECTION.md`](REFLECTION.md) (architecture, limitations, next steps).

## Design principles used throughout
- **Code computes facts; the LLM reasons over them.** Indicators, volatility, hedge price bands and data splits are
  deterministic code; the LLM classifies, weighs evidence and writes.
- **Every LLM output is validated** against a Pydantic schema; failures are logged and sent back once for repair, and
  numbers in generated reports are checked against the data they came from.
- **Prompts live in their own modules**, separate from business logic.
- **Tested offline:** 81 tests (scripted fake models drive the real code, so no API calls are needed).

## Repository layout
```
task1_financial/   data_pipeline.py, llm_reasoning.py, schemas.py, prompts.py, report.py, notebook, outputs/
task2_genai/       generate_data.py, dataset_report.py, finetune.py, evaluate.py, news_prompts.py, data/, notebook
task3_agentic/     tools.py, agent.py, multi_agent.py, memory.py, tracing.py, dashboard.py, logs/, notebook
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # add a free Groq key: GROQ_API_KEY=...
pytest task1_financial task2_genai task3_agentic
python task1_financial/report.py AAPL      # Task 1 brief  -> task1_financial/outputs/AAPL_brief.html
python task3_agentic/multi_agent.py MSFT   # Task 3 two-agent pipeline
streamlit run task3_agentic/dashboard.py   # Task 3 trace dashboard
```
Task 2 training needs a GPU: open its notebook in Colab with a T4 runtime and add `HF_TOKEN` (write access) and
`GROQ_API_KEY` as Colab Secrets. Keys are read from the environment or Colab Secrets only.

*Nothing in this repository is investment advice.*
