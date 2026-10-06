"""Task 2A - synthetic dataset generation for financial news impact analysis.

Pipeline (each stage is cached, so the script is resumable across Groq rate-limit windows):
  1. plan    - code builds balanced specs (event type x sector x direction x magnitude x style x company)
  2. generate- teacher (gpt-oss-120b) writes a fictional news item + reference analysis per spec
  3. label   - a second model (gpt-oss-20b) labels each item BLIND (text only, never the spec)
  4. build   - validate schema, check grounding, require teacher/blind agreement, drop near-duplicates,
               stratified 80/10/10 split, write chat-format JSONL + a diversity report

Run: python task2_genai/generate_data.py [--n 240] [--no-topup]
"""
import argparse
import json
import logging
import os
import random
import re
import statistics
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError
from pydantic import ValidationError

from news_prompts import (ANALYSIS_SYSTEM, ANALYSIS_USER, BLIND_LABEL_SYSTEM, BLIND_LABEL_USER, GENERATION_SYSTEM,
                     GENERATION_USER)
from news_schemas import EVENT_TYPES, MAGNITUDES, LabelFields, NewsAnalysis, grounding_issues

logger = logging.getLogger("generate_data")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CACHE_DIR = DATA_DIR / "cache"
GENERATED_CACHE = CACHE_DIR / "generated.jsonl"
LABEL_CACHE = CACHE_DIR / "blind_labels.jsonl"

# ---- Models (Groq free tier) ---------------------------------------------------------
TEACHER_MODEL = "openai/gpt-oss-120b"  # writes items + reference answers
LABELLER_MODEL = "openai/gpt-oss-20b"   # blind second opinion; separate quota from the teacher
GEN_TEMPERATURE = 0.9                   # variety in wording across items
LABEL_TEMPERATURE = 0.0                 # labelling should be deterministic
REASONING_EFFORT = "low"                # keeps token use inside the free-tier budget
TOKENS_PER_MINUTE = 8000                # Groq free-tier TPM for both models (from response headers)
MAX_ATTEMPTS = 3

# ---- Dataset plan ----------------------------------------------------------------------
N_SPECS = 240          # ~200 survive filtering -> ~160/20/20; brief minimum is 100
GEN_BATCH = 6          # specs per teacher call: amortises the long system prompt
LABEL_BATCH = 8        # items per blind-labelling call
SEED = 42
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
NEAR_DUP_HEADLINE, NEAR_DUP_SNIPPET = 0.75, 0.80   # word-level similarity at/above which two items are copies
# Same company + same event type is the same *scenario* at a much lower similarity (e.g. two buybacks by one firm)
SAME_SCENARIO_HEADLINE, SAME_SCENARIO_SNIPPET = 0.50, 0.40
HEADLINE_WORDS, SNIPPET_WORDS = (4, 20), (15, 90)  # lenient bounds around the prompt's 6-16 / 25-70

# Real, well-known companies (events are fictional). Region spread: US, Europe, Asia.
COMPANIES = {
    "technology": [("Apple Inc.", "AAPL", "US"), ("Microsoft Corp.", "MSFT", "US"), ("SAP SE", "SAP", "Europe"),
                   ("Sony Group Corp.", "6758.T", "Asia"), ("Adobe Inc.", "ADBE", "US")],
    "semiconductors": [("NVIDIA Corp.", "NVDA", "US"), ("Intel Corp.", "INTC", "US"), ("ASML Holding", "ASML", "Europe"),
                       ("Taiwan Semiconductor Manufacturing Co.", "TSM", "Asia"), ("Infineon Technologies", "IFX.DE", "Europe")],
    "banking": [("JPMorgan Chase & Co.", "JPM", "US"), ("Wells Fargo & Co.", "WFC", "US"), ("HSBC Holdings", "HSBA.L", "Europe"),
                ("Deutsche Bank AG", "DBK.DE", "Europe"), ("Mitsubishi UFJ Financial Group", "8306.T", "Asia")],
    "pharma": [("Pfizer Inc.", "PFE", "US"), ("Moderna Inc.", "MRNA", "US"), ("Novo Nordisk", "NVO", "Europe"),
               ("AstraZeneca plc", "AZN", "Europe"), ("Takeda Pharmaceutical", "4502.T", "Asia")],
    "energy": [("ExxonMobil", "XOM", "US"), ("Chevron Corp.", "CVX", "US"), ("Shell plc", "SHEL", "Europe"),
               ("BP plc", "BP", "Europe"), ("PetroChina", "0857.HK", "Asia")],
    "consumer_retail": [("Walmart Inc.", "WMT", "US"), ("Nike Inc.", "NKE", "US"), ("Unilever plc", "ULVR.L", "Europe"),
                        ("Adidas AG", "ADS.DE", "Europe"), ("Fast Retailing Co.", "9983.T", "Asia")],
    "autos": [("Ford Motor Co.", "F", "US"), ("Tesla Inc.", "TSLA", "US"), ("Volkswagen AG", "VOW3.DE", "Europe"),
              ("Toyota Motor Corp.", "7203.T", "Asia"), ("BYD Co.", "1211.HK", "Asia")],
    "industrials": [("Boeing Co.", "BA", "US"), ("Caterpillar Inc.", "CAT", "US"), ("Airbus SE", "AIR.PA", "Europe"),
                    ("Siemens AG", "SIE.DE", "Europe"), ("Mitsubishi Heavy Industries", "7011.T", "Asia")],
    "telecom_media": [("Verizon Communications", "VZ", "US"), ("Netflix Inc.", "NFLX", "US"), ("Walt Disney Co.", "DIS", "US"),
                      ("Vodafone Group", "VOD", "Europe"), ("SoftBank Group", "9984.T", "Asia")],
    "utilities": [("NextEra Energy", "NEE", "US"), ("Duke Energy", "DUK", "US"), ("Iberdrola SA", "IBE.MC", "Europe"),
                  ("Enel SpA", "ENEL.MI", "Europe"), ("Tokyo Electric Power", "9501.T", "Asia")],
}
STYLES = ("wire-service brief", "press-release excerpt", "analyst-note summary", "breaking-news alert")
# 40% positive / 40% negative / 20% neutral: neutral items are real but less informative to learn from
DIRECTION_MIX = ("positive", "positive", "negative", "negative", "neutral")


class DailyLimitReached(Exception):
    """Groq's per-day quota is exhausted; progress is cached, rerun later."""


# ======================================================================================
# 1. Plan
# ======================================================================================
def build_specs(n: int = N_SPECS, seed: int = SEED, event_plan: list[str] | None = None,
                id_offset: int = 0, stream: str = "") -> list[dict]:
    """Balanced specs: every dimension is drawn from shuffled full cycles, so each value
    appears (almost) equally often instead of whatever the teacher happens to favour.
    Each dimension has its own seeded RNG, so the plan is prefix-stable: spec i is the same
    for any n > i, and cached generations stay valid if --n changes.
    `event_plan` fixes the event types (used by the top-up); `stream` keeps its RNGs separate."""
    def cycle(values, name):
        rng = random.Random(f"{seed}-{stream}{name}")
        out = []
        while len(out) < n:
            block = list(values)
            rng.shuffle(block)
            out += block
        return out[:n]

    events = event_plan or cycle(EVENT_TYPES, "event")
    sectors, styles = cycle(COMPANIES, "sector"), cycle(STYLES, "style")
    directions, magnitudes = cycle(DIRECTION_MIX, "direction"), cycle(MAGNITUDES, "magnitude")
    company_cycles = {s: cycle(c, s) for s, c in COMPANIES.items()}  # rotate companies within each sector
    used = Counter()

    specs = []
    for i in range(n):
        sector = sectors[i]
        name, ticker, region = company_cycles[sector][used[sector]]
        used[sector] += 1
        direction = directions[i]
        specs.append({"spec_id": id_offset + i, "company": name, "ticker": ticker, "sector": sector,
                      "region": region, "event_type": events[i], "impact_direction": direction,
                      "magnitude": "low" if direction == "neutral" else magnitudes[i],  # labelling rule
                      "style": styles[i]})
    return specs


# Top-up: the first 240-spec pass kept only 41-59% of these four types (their labels are harder to
# make unambiguous, so the blind check rejects more). Extra specs, sized from that pass's yield, bring
# each to ~16 kept so every type has enough val/test examples. Fixed counts keep the plan reproducible.
TOPUP_PLAN = {"insider_transaction": 20, "management_change": 9, "macro": 8, "litigation": 8}
TOPUP_ID_OFFSET = 10_000  # far above any --n, so top-up ids never collide with the base plan


def topup_specs(seed: int = SEED) -> list[dict]:
    events = [e for e, k in TOPUP_PLAN.items() for _ in range(k)]
    random.Random(f"{seed}-topup-order").shuffle(events)
    return build_specs(len(events), seed, event_plan=events, id_offset=TOPUP_ID_OFFSET, stream="topup-")


# ======================================================================================
# 2-3. LLM calls (paced, retried, cached)
# ======================================================================================
def get_client() -> Groq:
    load_dotenv(HERE.parent / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set - add it to .env (see .env.example)")
    return Groq(max_retries=5)  # the SDK backs off on 429/5xx and honours retry-after


def chat_json(client: Groq, model: str, system: str, user: str, temperature: float) -> dict | None:
    """One JSON-mode call; returns the parsed object or None after MAX_ATTEMPTS. Sleeps after each
    call in proportion to tokens used, which keeps us under the per-minute token limit."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, reasoning_effort=REASONING_EFFORT,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        except RateLimitError as exc:
            if "per day" in str(exc).lower():
                raise DailyLimitReached(str(exc)) from exc
            logger.warning("rate limited (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc)
            time.sleep(60)
            continue
        except APIError as exc:
            logger.warning("API error (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc)
            continue

        time.sleep(resp.usage.total_tokens / TOKENS_PER_MINUTE * 60)
        try:
            return json.loads(resp.choices[0].message.content or "")
        except (json.JSONDecodeError, IndexError) as exc:
            logger.warning("invalid JSON from %s (attempt %d/%d): %s", model, attempt, MAX_ATTEMPTS, exc)
    return None


def _entries(out, key: str) -> list:
    """The list we asked for, whether the model wrapped it ({"examples": [...]}) or returned a
    bare list - both happen even in JSON mode. Anything else -> []."""
    if isinstance(out, list):
        return out
    if isinstance(out, dict) and isinstance(out.get(key), list):
        return out[key]
    return []


def _read_cache(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _append_cache(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def generate(client: Groq, specs: list[dict]) -> dict[int, dict]:
    """Teacher writes one item per spec. Returns {spec_id: raw item}. Cached per batch."""
    done = {item["spec_id"]: item for rec in _read_cache(GENERATED_CACHE) for item in rec["items"]}
    batches = [specs[i:i + GEN_BATCH] for i in range(0, len(specs), GEN_BATCH)]
    todo = [b for b in batches if not all(s["spec_id"] in done for s in b)]
    logger.info("generate: %d/%d batches cached, %d to go", len(batches) - len(todo), len(batches), len(todo))

    for k, batch in enumerate(todo, 1):
        user = GENERATION_USER.format(n=len(batch), specs_json="\n".join(json.dumps(s) for s in batch))
        out = chat_json(client, TEACHER_MODEL, GENERATION_SYSTEM, user, GEN_TEMPERATURE)
        wanted = {s["spec_id"] for s in batch}
        items = [it for it in _entries(out, "examples") if isinstance(it, dict) and it.get("spec_id") in wanted]
        if not items:
            logger.warning("batch %s produced no usable items", sorted(wanted))
            continue
        _append_cache(GENERATED_CACHE, {"spec_ids": sorted(wanted), "items": items})
        done.update({it["spec_id"]: it for it in items})
        logger.info("generate: batch %d/%d -> %d items (total %d)", k, len(todo), len(items), len(done))
    return done


def news_text(item: dict) -> str:
    return ANALYSIS_USER.format(headline=item.get("headline", ""), snippet=item.get("snippet", ""))


def blind_label(client: Groq, generated: dict[int, dict]) -> dict[int, dict]:
    """Labeller sees only headline + snippet (never the spec). Returns {spec_id: raw analysis}."""
    done = {rec["id"]: rec["analysis"] for rec in _read_cache(LABEL_CACHE)}
    ids = [i for i in sorted(generated) if i not in done
           and isinstance(generated[i].get("headline"), str) and isinstance(generated[i].get("snippet"), str)]
    logger.info("label: %d cached, %d to go", len(done), len(ids))

    for k in range(0, len(ids), LABEL_BATCH):
        chunk = ids[k:k + LABEL_BATCH]
        items = [{"id": i, "headline": generated[i]["headline"], "snippet": generated[i]["snippet"]} for i in chunk]
        user = BLIND_LABEL_USER.format(n=len(items), items_json="\n".join(json.dumps(x) for x in items))
        out = chat_json(client, LABELLER_MODEL, BLIND_LABEL_SYSTEM, user, LABEL_TEMPERATURE)
        for a in _entries(out, "analyses"):
            if isinstance(a, dict) and a.get("id") in chunk:
                _append_cache(LABEL_CACHE, {"id": a.pop("id"), "analysis": a})
        logger.info("label: %d/%d", min(k + LABEL_BATCH, len(ids)), len(ids))
    return {rec["id"]: rec["analysis"] for rec in _read_cache(LABEL_CACHE)}


# ======================================================================================
# 4. Build: filter, dedupe, split, write
# ======================================================================================
def _words(s) -> int:
    return len(str(s).split())


# The teacher sometimes writes the spec's *style* into the text ("A breaking-news alert reported that ...").
# Real news never describes itself that way, so these phrases are stripped. Normal journalism such as
# "Ford issued a press release confirming ..." is deliberately left alone.
_SEP = r"[\s\-‑]?"  # space, hyphen or non-breaking hyphen
_STYLE = (rf"(?:press{_SEP}release{_SEP}excerpt|breaking{_SEP}news{_SEP}alert|wire{_SEP}service{_SEP}brief|"
          rf"analyst{_SEP}note(?:{_SEP}summary)?)")
_STYLE_LEAKS = [
    # "In a breaking-news alert, X ..." / "In its latest analyst note, X ..." / "In a wire-service brief dated 9 May, X"
    re.compile(rf"^In (?:a|an|its latest) (?:brief )?{_STYLE}(?: (?:released|dated|on) [^,]+)?,\s*", re.I),
    # "A wire-service brief noted that X ..." / "A breaking-news alert on 14 May reported that X ..."
    re.compile(rf"^(?:A|An) {_STYLE}(?: (?:on|dated) [^,]+?)? (?:reported|noted|detailed|summari[sz]es|said|stated)"
               rf"(?: that)?\s+", re.I),
    # "... was disclosed in an analyst-note summary on 3 May."
    re.compile(rf"\s+in an? {_STYLE}(?=[\s.,])", re.I),
    # headline prefix "Analyst note: ..."
    re.compile(r"^(?:analyst note|breaking(?: news)?|press release)\s*:\s*", re.I),
]


def strip_style_leaks(text: str) -> tuple[str, bool]:
    out = text
    for pattern in _STYLE_LEAKS:
        out = pattern.sub("", out)
    out = out.strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out, out != text.strip()


def filter_examples(specs: list[dict], generated: dict[int, dict],
                    labels: dict[int, dict]) -> tuple[list[dict], Counter]:
    """Keep an item only if it passes every check. Returns (kept, rejection reasons).
    Checks, in order: present -> lengths -> schema -> teacher matches spec -> grounded ->
    blind labeller agrees (same event_type and direction, magnitude within one step)."""
    kept, rejected = [], Counter()
    for spec in specs:
        item = generated.get(spec["spec_id"])
        if item is None:
            rejected["not_generated"] += 1
            continue
        h, s = item.get("headline"), item.get("snippet")
        fixes = []
        if isinstance(h, str) and isinstance(s, str):
            (h, h_fixed), (s, s_fixed) = strip_style_leaks(h), strip_style_leaks(s)
            if h_fixed or s_fixed:
                fixes.append("style_phrase_removed")
            item = {**item, "headline": h, "snippet": s}  # grounding below runs on the cleaned text
        if not (isinstance(h, str) and isinstance(s, str)
                and HEADLINE_WORDS[0] <= _words(h) <= HEADLINE_WORDS[1]
                and SNIPPET_WORDS[0] <= _words(s) <= SNIPPET_WORDS[1]):
            rejected["length_or_missing_text"] += 1
            continue
        try:
            ref = NewsAnalysis.model_validate(item.get("analysis"))
        except ValidationError:
            rejected["schema_invalid"] += 1
            continue
        if (ref.event_type, ref.impact_direction, ref.magnitude) != (
                spec["event_type"], spec["impact_direction"], spec["magnitude"]):
            rejected["teacher_off_spec"] += 1
            continue
        text = news_text(item)
        if grounding_issues(text, ref):
            rejected["ungrounded"] += 1
            continue
        try:
            blind = LabelFields.model_validate(labels.get(spec["spec_id"]))
        except ValidationError:
            rejected["blind_label_missing_or_invalid"] += 1
            continue
        mag_gap = abs(MAGNITUDES.index(blind.magnitude) - MAGNITUDES.index(ref.magnitude))
        if blind.event_type != ref.event_type or blind.impact_direction != ref.impact_direction or mag_gap > 1:
            rejected["blind_disagrees"] += 1
            continue
        # Repair: the teacher sometimes omits the main company from mentioned_entities. Grounding only
        # checks listed names are in the text, not that every name is listed, so add it back here.
        # (macro: only the spec's company, never a sector label like "energy sector".)
        main = ref.primary_entity if ref.event_type != "macro" else spec["company"]
        listed = {e.strip().lower() for e in ref.mentioned_entities}
        if main.lower() in text.lower() and main.lower() not in listed:
            ref = ref.model_copy(update={"mentioned_entities": [main, *ref.mentioned_entities]})
            fixes.append("primary_entity_added_to_mentioned")
        kept.append({**spec, "headline": h.strip(), "snippet": s.strip(), "input": text,
                     "output": ref.model_dump(), "blind_magnitude_matches": mag_gap == 0, "fixes": fixes})
    return kept, rejected


def similarity(a: str, b: str) -> float:
    """Word-level similarity in [0, 1]. autojunk=False: difflib's default silently ignores
    'popular' items in sequences of 200+ elements, which made character-level scores jump
    depending on text length."""
    return SequenceMatcher(None, a.lower().split(), b.lower().split(), autojunk=False).ratio()


def drop_near_duplicates(examples: list[dict]) -> tuple[list[dict], int]:
    """Greedy: drop an item that is a near-copy of one already kept, or the same scenario (same
    company and event type with similar wording). Similar headline *templates* across different
    companies ("X beats forecast with 12% revenue growth") are normal newswire style and are kept."""
    def duplicate(ex, k):
        h_sim, s_sim = similarity(ex["headline"], k["headline"]), similarity(ex["snippet"], k["snippet"])
        if h_sim >= NEAR_DUP_HEADLINE or s_sim >= NEAR_DUP_SNIPPET:
            return True
        same_scenario = ex.get("company") == k.get("company") and ex.get("event_type") == k.get("event_type")
        return same_scenario and (h_sim >= SAME_SCENARIO_HEADLINE or s_sim >= SAME_SCENARIO_SNIPPET)

    kept = []
    for ex in examples:
        if not any(duplicate(ex, k) for k in kept):
            kept.append(ex)
    return kept, len(examples) - len(kept)


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Share `total` slots across groups in proportion to their size (largest-remainder method),
    with at least 1 per group of 3+ items. Rounding globally instead of per group keeps the
    overall split at 80/10/10 - per-group rounding of 12-21 items turned 10% into ~11%."""
    eligible = {g: n for g, n in sizes.items() if n >= 3}
    n_all = sum(sizes.values())
    quota = {g: n * total / n_all for g, n in eligible.items()}
    alloc = {g: max(1, int(q)) for g, q in quota.items()}
    by_remainder = sorted(eligible, key=lambda g: quota[g] - int(quota[g]), reverse=True)
    for g in by_remainder[:max(0, total - sum(alloc.values()))]:
        alloc[g] += 1
    return {g: alloc.get(g, 0) for g in sizes}


def stratified_split(examples: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    """80/10/10 overall, stratified by event_type so every type appears in val and test."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        by_type.setdefault(ex["event_type"], []).append(ex)
    for group in by_type.values():
        rng.shuffle(group)

    sizes = {t: len(g) for t, g in by_type.items()}
    n_test = allocate(sizes, round(len(examples) * SPLIT_FRACTIONS[2]))
    n_val = allocate(sizes, round(len(examples) * SPLIT_FRACTIONS[1]))
    splits = {"train": [], "val": [], "test": []}
    for t, group in by_type.items():
        splits["test"] += group[:n_test[t]]
        splits["val"] += group[n_test[t]:n_test[t] + n_val[t]]
        splits["train"] += group[n_test[t] + n_val[t]:]
    for part in splits.values():
        rng.shuffle(part)
    return splits


def to_chat_record(ex: dict) -> dict:
    """System / user / assistant turns. The model's chat template is applied at training time
    (tokenizer.apply_chat_template), which is how TRL's SFTTrainer consumes 'messages'."""
    return {"messages": [{"role": "system", "content": ANALYSIS_SYSTEM},
                         {"role": "user", "content": ex["input"]},
                         {"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)}]}


STOPWORDS = set("""a an the and or of to in on for at by with from as is are was were be been its it this that
than after before over into amid while said says will would could per year quarter shares share company
inc corp co plc ag se sa ltd group holdings percent million billion new also which has have had not
their they about more up down""".split())


def diversity_report(splits: dict[str, list[dict]]) -> dict:
    """Evidence that the dataset is not one scenario reworded: label/sector/style balance,
    prompt-length distribution, keyword frequencies, lexical diversity, nearest-neighbour similarity."""
    allx = [ex for part in splits.values() for ex in part]
    if not allx:
        return {"split_sizes": {k: 0 for k in splits}, "total": 0}
    lengths = [_words(ex["input"]) for ex in allx]

    company_tokens = {t.lower().strip(".,") for names in COMPANIES.values() for n, _, _ in names for t in n.split()}
    tokens = [t for ex in allx for t in re.findall(r"[a-z][a-z\-]+", ex["input"].lower())
              if t not in STOPWORDS and t not in company_tokens and t not in ("headline", "snippet") and len(t) > 2]

    bigrams = [b for ex in allx for b in zip(ex["input"].lower().split(), ex["input"].lower().split()[1:])]
    nearest = []
    texts = [ex["snippet"].lower() for ex in allx]
    for i, t in enumerate(texts):
        nearest.append(max((similarity(t, u) for j, u in enumerate(texts) if j != i), default=0))

    q = statistics.quantiles(lengths, n=4, method="inclusive") if len(lengths) >= 2 else [0, 0, 0]
    return {
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "total": len(allx),
        "counts": {dim: dict(Counter(ex[dim] for ex in allx).most_common())
                   for dim in ("event_type", "sector", "region", "impact_direction", "magnitude", "style")},
        "unique_companies": len({ex["company"] for ex in allx}),
        "prompt_length_words": {"min": min(lengths), "p25": q[0], "median": q[1], "p75": q[2], "max": max(lengths),
                                "mean": round(statistics.mean(lengths), 1), "values": lengths},
        "top_keywords": Counter(tokens).most_common(30),
        "distinct_2": round(len(set(bigrams)) / len(bigrams), 3),  # unique / total word bigrams
        "nearest_neighbour_similarity": {"mean": round(statistics.mean(nearest), 3),
                                         "p95": round(statistics.quantiles(nearest, n=20, method="inclusive")[-1], 3),
                                         "max": round(max(nearest), 3)},
    }


def build(specs: list[dict], generated: dict[int, dict], labels: dict[int, dict]) -> dict:
    kept, rejected = filter_examples(specs, generated, labels)
    kept, n_dups = drop_near_duplicates(kept)
    rejected["near_duplicate"] = n_dups
    splits = stratified_split(kept)

    for name, part in splits.items():
        (DATA_DIR / f"{name}.jsonl").write_text("".join(json.dumps(to_chat_record(ex), ensure_ascii=False) + "\n"
                                                        for ex in part))
    with (DATA_DIR / "examples_with_metadata.jsonl").open("w") as f:
        for name, part in splits.items():
            for ex in part:
                f.write(json.dumps({"split": name, **ex}, ensure_ascii=False) + "\n")

    report = diversity_report(splits)
    report["planned_specs"] = len(specs)
    report["rejected"] = dict(rejected)
    report["blind_magnitude_exact_agreement"] = round(
        sum(ex["blind_magnitude_matches"] for ex in kept) / max(1, len(kept)), 3)
    report["automatic_fixes"] = dict(Counter(f for ex in kept for f in ex.get("fixes", [])))
    (DATA_DIR / "diversity_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(n: int, topup: bool = True) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(DATA_DIR / "generation.log")])
    logging.getLogger("httpx").setLevel(logging.WARNING)

    specs = build_specs(n) + (topup_specs() if topup else [])
    (DATA_DIR / "specs.json").write_text(json.dumps(specs, indent=1))
    client = get_client()
    try:
        generated = generate(client, specs)
        labels = blind_label(client, generated)
    except DailyLimitReached as exc:
        logger.error("Groq daily quota reached - progress is cached, rerun later to continue. %s", exc)
        return

    report = build(specs, generated, labels)
    print(json.dumps({k: report.get(k) for k in ("planned_specs", "split_sizes", "rejected",
                                             "blind_magnitude_exact_agreement", "distinct_2",
                                             "nearest_neighbour_similarity")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=N_SPECS, help="number of balanced specs to plan")
    parser.add_argument("--no-topup", action="store_true", help="skip the top-up specs for hard event types")
    args = parser.parse_args()
    main(args.n, topup=not args.no_topup)
