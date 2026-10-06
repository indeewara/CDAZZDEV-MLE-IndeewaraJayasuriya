"""Task 2A evidence, generated from the actual dataset files (nothing typed by hand):
  - validates every JSONL record (3 chat turns, exact system prompt, schema-valid and grounded target)
  - renders one record in Qwen2.5's real chat template and measures token lengths
  - draws the diversity charts (prompt lengths, topics, keywords, similarity)
  - writes TEACHER_PROMPTS.md verbatim from news_prompts.py

Run: python task2_genai/dataset_report.py
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from news_prompts import (ANALYSIS_SYSTEM, ANALYSIS_USER, BLIND_LABEL_SYSTEM, BLIND_LABEL_USER, GENERATION_SYSTEM,
                          GENERATION_USER)
from news_schemas import NewsAnalysis, grounding_issues

HERE = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR = HERE / "data", HERE / "outputs"
SPLITS = ("train", "val", "test")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # student; the 3B variant uses the identical tokenizer/template

# chart styling: one categorical colour (single series per panel), recessive axes
BAR, INK_MUTED, GRID = "#2a78d6", "#52514e", "#ecebe7"


def load_split(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]


def validate_records(records: list[dict]) -> list[str]:
    """Every record: system/user/assistant turns, the exact training system prompt, and an
    assistant target that passes the schema and the grounding check. Returns problems found."""
    problems = []
    for i, rec in enumerate(records):
        msgs = rec.get("messages", [])
        if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
            problems.append(f"record {i}: roles {[m.get('role') for m in msgs]}")
            continue
        if msgs[0]["content"] != ANALYSIS_SYSTEM:
            problems.append(f"record {i}: system prompt differs from ANALYSIS_SYSTEM")
        try:
            target = NewsAnalysis.model_validate_json(msgs[2]["content"])
        except Exception as exc:  # noqa: BLE001 - report any parse/validation failure
            problems.append(f"record {i}: invalid target ({exc})")
            continue
        problems += [f"record {i}: {p}" for p in grounding_issues(msgs[1]["content"], target)]
    return problems


def get_tokenizer():
    from transformers import AutoTokenizer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AutoTokenizer.from_pretrained(BASE_MODEL)


def _style(ax, title, grid_axis="y"):
    ax.set_title(title, loc="left", fontsize=10, color=INK_MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _hbar(ax, counts: dict, title: str, sort: bool = True):
    items = sorted(counts.items(), key=lambda kv: kv[1]) if sort else list(counts.items())[::-1]
    ax.barh([k for k, _ in items], [v for _, v in items], color=BAR, height=0.7)
    for y, (_, v) in enumerate(items):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=7, color=INK_MUTED)
    _style(ax, title, grid_axis="x")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def make_charts(report: dict, token_lengths: list[int], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    (a, b, c), (d, e, f) = axes

    a.hist(report["prompt_length_words"]["values"], bins=12, color=BAR, edgecolor="white")
    _style(a, "Input length (words, headline + snippet)")
    a.set_xlabel("words", color=INK_MUTED, fontsize=8)

    _hbar(b, report["counts"]["event_type"], "Topic: event type")
    _hbar(c, dict(report["top_keywords"][:15]), "Top 15 keywords (excl. stopwords, company names)")

    _hbar(d, report["counts"]["sector"], "Topic: sector")

    labels = [f"direction: {k}" for k in report["counts"]["impact_direction"]] + \
             [f"magnitude: {k}" for k in report["counts"]["magnitude"]] + \
             [f"region: {k}" for k in report["counts"]["region"]]
    values = list(report["counts"]["impact_direction"].values()) + list(report["counts"]["magnitude"].values()) + \
        list(report["counts"]["region"].values())
    _hbar(e, dict(zip(labels, values)), "Direction, magnitude and region", sort=False)

    f.hist(token_lengths, bins=12, color=BAR, edgecolor="white")
    _style(f, "Training sequence length (Qwen2.5 tokens)")
    f.set_xlabel("tokens (system + user + assistant, chat template applied)", color=INK_MUTED, fontsize=8)

    fig.suptitle("Task 2A dataset diversity", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_teacher_prompts(path: Path) -> None:
    """Verbatim copy of every prompt used to build the dataset, for the submission."""
    sections = [
        ("1. Teacher - data generation (`openai/gpt-oss-120b`)", "System prompt", GENERATION_SYSTEM,
         "User prompt template (`{specs_json}` = one JSON spec per line, 6 per call)", GENERATION_USER),
        ("2. Blind labeller - quality filter (`openai/gpt-oss-20b`)", "System prompt", BLIND_LABEL_SYSTEM,
         "User prompt template (8 items per call, text only - never the spec)", BLIND_LABEL_USER),
        ("3. Training / inference system prompt (the `system` turn of every JSONL record)", "System prompt",
         ANALYSIS_SYSTEM, "User turn template", ANALYSIS_USER),
    ]
    lines = ["# Task 2 - Prompts used to build the dataset", "",
             "Generated verbatim from `news_prompts.py` by `dataset_report.py`, so this file always matches the code.",
             "All three prompts embed the same `TASK_RULES` block (taxonomy, field definitions, labelling rules).", ""]
    for title, l1, p1, l2, p2 in sections:
        lines += [f"## {title}", "", f"**{l1}**", "", "```text", p1, "```", "", f"**{l2}**", "", "```text", p2, "```", ""]
    path.write_text("\n".join(lines))


def main() -> dict:
    report = json.loads((DATA_DIR / "diversity_report.json").read_text())
    splits = {s: load_split(s) for s in SPLITS}
    problems = {s: validate_records(r) for s, r in splits.items()}

    tok = get_tokenizer()
    all_records = [r for recs in splits.values() for r in recs]
    # render to text, then tokenize: newer transformers return a dict from tokenize=True
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in all_records]
    token_lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    rendered = texts[0]
    (DATA_DIR / "sample_chat_template.txt").write_text(rendered)

    make_charts(report, token_lengths, OUT_DIR / "dataset_diversity.png")
    write_teacher_prompts(HERE / "TEACHER_PROMPTS.md")

    n = sum(len(r) for r in splits.values())
    summary = {
        "split_sizes": {s: f"{len(r)} ({len(r) / n:.1%})" for s, r in splits.items()},
        "total": n,
        "jsonl_problems": {s: len(p) for s, p in problems.items()},
        "token_length": {"min": min(token_lengths), "max": max(token_lengths),
                         "mean": round(sum(token_lengths) / n)},
    }
    return {"summary": summary, "problems": problems, "rendered_example": rendered}


if __name__ == "__main__":
    out = main()
    print(json.dumps(out["summary"], indent=2))
    for split, probs in out["problems"].items():
        for p in probs[:5]:
            print(f"[{split}] {p}")
    print("\n--- one training record rendered with the Qwen2.5 chat template (truncated) ---")
    print(out["rendered_example"][:400] + "\n...\n" + out["rendered_example"][-420:])
