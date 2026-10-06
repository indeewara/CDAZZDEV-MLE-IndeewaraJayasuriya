"""Task 2C - base vs fine-tuned evaluation on the held-out test set.

Both models get the identical system prompt (ANALYSIS_SYSTEM) and greedy decoding. Metrics:
  - format: strict schema validity (exactly the six keys, allowed values)
  - labels: event_type / direction / magnitude / primary_entity accuracy, read leniently (any JSON object),
    so a model that gets labels right but breaks the format is not penalised twice
  - text: ROUGE-L on the rationale (required) and on the full output; BERTScore F1 on the rationale
  - grounding: automatic check that every name and number is in the input (hallucination proxy)
  - LLM-as-judge: qwen/qwen3.8-27b scores a rubric and returns validated JSON
  - manual review: a sheet for hand-labelling the fine-tuned outputs (hallucination rate)
"""
import csv
import json
import os
import re
import time
from pathlib import Path

from pydantic import ValidationError

from news_prompts import ANALYSIS_SYSTEM, JUDGE_SYSTEM, JUDGE_USER
from news_schemas import VERDICTS, JudgeScore, NewsAnalysis, _normalise_label, grounding_issues

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

MAX_NEW_TOKENS = 300      # reference answers are ~80-120 tokens; 300 leaves room without letting a model ramble forever
GEN_BATCH_SIZE = 6        # 18 test items -> 3 batches per model on a T4
JUDGE_MODEL = "qwen/qwen3.8-27b"
# Why this judge: a different family from the teacher (OpenAI gpt-oss), so it has no preference for the
# teacher's phrasing that the fine-tuned model has learned. It shares a family with the student (Qwen2.5),
# which if anything favours the *base* model's own style - a conservative bias against our claimed improvement.
JUDGE_TOKENS_PER_MINUTE = 8000
LABEL_FIELDS = ("event_type", "impact_direction", "magnitude", "primary_entity")
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


# ======================================================================================
# Generation (GPU, in the notebook)
# ======================================================================================
def generate(model, tokenizer, prompts: list[list[dict]], batch_size: int = GEN_BATCH_SIZE,
             max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
    """Greedy decoding (deterministic, comparable across models) with left padding for batched generation."""
    import torch
    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    outputs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outputs


# ======================================================================================
# Parsing and per-item scoring (pure functions, unit-tested)
# ======================================================================================
def lenient_json(raw: str) -> dict | None:
    """The first {...} object in the output, even if wrapped in prose or a code fence; None if absent."""
    m = _JSON_OBJECT.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def strict_parse(raw: str) -> NewsAnalysis | None:
    """Schema-valid analysis, or None. The output must be essentially just the JSON object."""
    text = (raw or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        return NewsAnalysis.model_validate_json(text)
    except ValidationError:
        return None


def _norm(field: str, value) -> str:
    if value is None:
        return ""
    return _normalise_label(str(value)) if field != "primary_entity" else str(value).strip().lower()


def score_item(raw: str, reference: dict, news: str) -> dict:
    """Per-item results for one model output against the reference analysis."""
    loose, strict = lenient_json(raw), strict_parse(raw)
    row = {"schema_valid": strict is not None, "json_found": loose is not None,
           "rationale": str((loose or {}).get("rationale", "")), "raw": raw}
    for f in LABEL_FIELDS:
        row[f"{f}_correct"] = loose is not None and _norm(f, loose.get(f)) == _norm(f, reference[f])
    row["grounding_issues"] = grounding_issues(news, strict) if strict else None
    row["grounded"] = strict is not None and not row["grounding_issues"]
    return row


# ======================================================================================
# Text metrics
# ======================================================================================
def rouge_l(candidates: list[str], references: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, cand)["rougeL"].fmeasure for cand, ref in zip(candidates, references)]


def bertscore_f1(candidates: list[str], references: list[str]) -> list[float]:
    """BERTScore F1 with roberta-large (the library's English default), baseline-rescaled so that
    unrelated sentences score ~0 instead of ~0.8, which makes differences readable."""
    from bert_score import score
    cands = [c if c.strip() else "(empty)" for c in candidates]  # an empty candidate would crash the scorer
    _, _, f1 = score(cands, references, lang="en", rescale_with_baseline=True, verbose=False)
    return [float(x) for x in f1]


# ======================================================================================
# LLM-as-judge
# ======================================================================================
def judge(client, news: str, reference: dict, candidate_raw: str, max_attempts: int = 3) -> JudgeScore | None:
    """One rubric score, validated with Pydantic; None (logged by the caller) if every attempt fails."""
    user = JUDGE_USER.format(news=news, reference=json.dumps(reference, ensure_ascii=False),
                             candidate=candidate_raw.strip() or "(empty output)")
    for _ in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, reasoning_effort="none", response_format={"type": "json_object"},
                messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])
            time.sleep(resp.usage.total_tokens / JUDGE_TOKENS_PER_MINUTE * 60)  # stay under the free-tier TPM
            return JudgeScore.model_validate_json(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 - API or validation failure: retry, then give up gracefully
            print(f"judge attempt failed: {type(exc).__name__}: {str(exc)[:150]}")
    return None


# ======================================================================================
# Aggregation
# ======================================================================================
def summarise(rows: list[dict], rouge_rat: list[float], rouge_full: list[float], bert: list[float] | None,
              judged: list[JudgeScore | None] | None) -> dict:
    n = len(rows)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    out = {
        "schema_valid_%": round(100 * sum(r["schema_valid"] for r in rows) / n, 1),
        **{f"{f}_acc_%": round(100 * sum(r[f"{f}_correct"] for r in rows) / n, 1) for f in LABEL_FIELDS},
        "grounded_%": round(100 * sum(r["grounded"] for r in rows) / n, 1),
        "rougeL_rationale": mean(rouge_rat),
        "rougeL_full_output": mean(rouge_full),
        "bertscore_f1_rationale": mean(bert) if bert is not None else None,
    }
    if judged is not None:
        ok = [j for j in judged if j is not None]
        for k in ("label_accuracy", "grounding", "rationale_quality", "format_compliance"):
            out[f"judge_{k}"] = mean([getattr(j, k) for j in ok])
        out.update({f"judge_{v}_%": round(100 * sum(j.verdict == v for j in ok) / max(1, len(ok)), 1) for v in VERDICTS})
        out["judge_failures"] = len(judged) - len(ok)
    return out


def paired_wins(a: list[float], b: list[float]) -> dict:
    """Per-item comparison of model b against model a (e.g. fine-tuned vs base ROUGE-L)."""
    return {"b_better": sum(y > x for x, y in zip(a, b)), "tie": sum(y == x for x, y in zip(a, b)),
            "a_better": sum(y < x for x, y in zip(a, b))}


# ======================================================================================
# Manual review sheet
# ======================================================================================
REVIEW_FIELDS = ["id", "news", "reference", "fine_tuned_output", "auto_grounding_issues", "judge_verdict",
                 "manual_label", "notes"]


def write_review_sheet(path: Path, news: list[str], refs: list[dict], ft_raw: list[str], ft_rows: list[dict],
                       ft_judged: list[JudgeScore | None]) -> None:
    """CSV for hand-labelling every fine-tuned test output as correct / partially_correct / hallucinated
    (or incorrect). manual_label is left blank on purpose - the reviewer fills it in."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for i, (n, ref, raw, row, j) in enumerate(zip(news, refs, ft_raw, ft_rows, ft_judged)):
            w.writerow({"id": i, "news": n, "reference": json.dumps(ref, ensure_ascii=False), "fine_tuned_output": raw,
                        "auto_grounding_issues": "; ".join(row["grounding_issues"] or []) if row["schema_valid"] else "not schema-valid",
                        "judge_verdict": j.verdict if j else "", "manual_label": "", "notes": ""})


def hallucination_rate(path: Path) -> dict | None:
    """From a filled-in review sheet: label counts and hallucination rate. None until labels are filled."""
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    labels = [r["manual_label"].strip().lower() for r in rows if r["manual_label"].strip()]
    if not labels:
        return None
    unknown = sorted(set(labels) - set(VERDICTS))
    if unknown:
        raise ValueError(f"unknown manual labels {unknown}; use one of {VERDICTS}")
    counts = {v: labels.count(v) for v in VERDICTS}
    return {"reviewed": len(labels), "counts": counts,
            "hallucination_rate_%": round(100 * counts["hallucinated"] / len(labels), 1)}


def groq_client():
    from dotenv import load_dotenv
    from groq import Groq
    load_dotenv(HERE.parent / ".env")
    return Groq(max_retries=5) if os.getenv("GROQ_API_KEY") else None
