"""Offline tests for Task 2A data generation: plan balance, schema, grounding, filtering, split, format."""
import json
from collections import Counter

import pytest
from pydantic import ValidationError

from generate_data import (build_specs, drop_near_duplicates, filter_examples, news_text, stratified_split,
                           to_chat_record)
from news_prompts import ANALYSIS_SYSTEM
from news_schemas import EVENT_TYPES, NewsAnalysis, grounding_issues

GOOD = {"event_type": "guidance", "primary_entity": "Nike Inc.", "mentioned_entities": ["Nike Inc."],
        "impact_direction": "negative", "magnitude": "high",
        "rationale": "Nike cut its growth forecast from 5% to 2%. A guidance cut resets expectations for the whole year."}
ITEM = {"headline": "Nike Inc. cuts full-year revenue outlook as China demand slows",
        "snippet": "Nike Inc. lowered its fiscal-year revenue growth forecast to 2% from 5%, citing weaker footwear "
                   "demand in Greater China, and said margins would also come under pressure."}


def test_specs_are_balanced_prefix_stable_and_follow_neutral_rule():
    specs = build_specs(220)
    events = Counter(s["event_type"] for s in specs)
    assert set(events) == set(EVENT_TYPES) and max(events.values()) - min(events.values()) <= 1
    assert all(s["magnitude"] == "low" for s in specs if s["impact_direction"] == "neutral")
    assert build_specs(10) == specs[:10]  # cache-safe when --n changes


def test_schema_normalises_and_rejects():
    assert NewsAnalysis.model_validate({**GOOD, "event_type": "M&A"}).event_type == "m_and_a"
    for bad in [{**GOOD, "extra": 1}, {**GOOD, "event_type": "merger"}, {**GOOD, "magnitude": "huge"},
                {**GOOD, "rationale": "Too short."}, {k: v for k, v in GOOD.items() if k != "rationale"}]:
        with pytest.raises(ValidationError):
            NewsAnalysis.model_validate(bad)


def test_grounding_flags_invented_entities_and_numbers():
    text = news_text(ITEM)
    assert grounding_issues(text, NewsAnalysis.model_validate(GOOD)) == []
    invented = NewsAnalysis.model_validate({**GOOD, "mentioned_entities": ["Nike Inc.", "Adidas AG"],
                                            "rationale": "Nike cut guidance to 2%. Shares fell 9% after hours."})
    issues = grounding_issues(text, invented)
    assert any("Adidas" in i for i in issues) and any("9" in i for i in issues)


def _spec(i, **kw):
    return {"spec_id": i, "company": "Nike Inc.", "ticker": "NKE", "sector": "consumer_retail", "region": "US",
            "event_type": "guidance", "impact_direction": "negative", "magnitude": "high", "style": "x", **kw}


def test_filter_reasons():
    specs = [_spec(i) for i in range(6)]
    generated = {
        0: {**ITEM, "analysis": GOOD},                                                  # kept
        1: {**ITEM, "analysis": {**GOOD, "magnitude": "low"}},                          # teacher off spec
        2: {**ITEM, "analysis": {**GOOD, "mentioned_entities": ["Adidas AG"]}},         # ungrounded
        3: {**ITEM, "analysis": GOOD},                                                  # blind disagrees
        4: {"headline": "Too short", "snippet": ITEM["snippet"], "analysis": GOOD},     # length
    }                                                                                   # 5: not generated
    labels = {0: {**GOOD, "magnitude": "medium"}, 1: GOOD, 2: GOOD,
              3: {**GOOD, "impact_direction": "positive"}, 4: GOOD}
    kept, rejected = filter_examples(specs, generated, labels)
    assert [k["spec_id"] for k in kept] == [0] and kept[0]["blind_magnitude_matches"] is False
    assert rejected == Counter(teacher_off_spec=1, ungrounded=1, blind_disagrees=1,
                               length_or_missing_text=1, not_generated=1)


def test_near_duplicates_dropped():
    a = {"headline": "Nike cuts outlook on weak China demand", "snippet": "Nike lowered guidance to 2% growth."}
    b = {"headline": "Nike cuts outlook on weak China sales", "snippet": "Nike lowered guidance to 2% growth now."}
    c = {"headline": "Boeing wins 40-jet order from Gulf carrier", "snippet": "Boeing booked a large widebody order."}
    kept, n = drop_near_duplicates([a, b, c])
    assert kept == [a, c] and n == 1


def test_stratified_split_covers_every_type_in_test():
    examples = [{"event_type": t, "id": f"{t}{i}"} for t in EVENT_TYPES for i in range(20)]
    splits = stratified_split(examples)
    assert {k: len(v) for k, v in splits.items()} == {"train": 176, "val": 22, "test": 22}
    assert {e["event_type"] for e in splits["test"]} == set(EVENT_TYPES)
    assert not {e["id"] for e in splits["test"]} & {e["id"] for e in splits["train"]}


def test_chat_record_has_three_turns_and_valid_target():
    rec = to_chat_record({"input": news_text(ITEM), "output": GOOD})
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == ANALYSIS_SYSTEM
    NewsAnalysis.model_validate(json.loads(rec["messages"][2]["content"]))


def test_blind_label_only_needs_the_three_label_fields():
    specs = [_spec(0)]
    one_sentence_blind = {"event_type": "Guidance", "impact_direction": "negative", "magnitude": "high",
                          "rationale": "Cut outlook."}
    kept, rejected = filter_examples(specs, {0: {**ITEM, "analysis": GOOD}}, {0: one_sentence_blind})
    assert len(kept) == 1 and not rejected


def test_topup_specs_only_hard_types_and_never_collide():
    from generate_data import TOPUP_ID_OFFSET, TOPUP_PLAN, topup_specs
    top = topup_specs()
    assert Counter(s["event_type"] for s in top) == Counter(TOPUP_PLAN)
    assert min(s["spec_id"] for s in top) >= TOPUP_ID_OFFSET
    assert top == topup_specs()                          # reproducible
    assert build_specs(240)[:5] == build_specs(5)        # base plan untouched by the refactor


@pytest.mark.parametrize("raw, expected", [
    ("A breaking‑news alert reported that Tesla Inc. cut prices by 5%.", "Tesla Inc. cut prices by 5%."),
    ("A wire-service brief noted that the Federal Reserve held rates.", "The Federal Reserve held rates."),
    ("A breaking-news alert on 14 May reported that Ford recalled 3,000 trucks.", "Ford recalled 3,000 trucks."),
    ("In a press‑release excerpt, Toyota announced a 4% dividend rise.", "Toyota announced a 4% dividend rise."),
    ("In an analyst‑note summary released 10 May, Shell raised its target.", "Shell raised its target."),
    ("In its latest analyst note, Iberdrola announced a buyback.", "Iberdrola announced a buyback."),
    ("The cut was disclosed in an analyst‑note summary on 3 May.", "The cut was disclosed on 3 May."),
    ("Analyst note: New drug-price framework boosts pharma", "New drug-price framework boosts pharma"),
    ("In a brief analyst note, Morgan Stanley downgraded BP.", "Morgan Stanley downgraded BP."),  # lossless
    # normal journalism is left alone
    ("Ford Motor Co. issued a press release confirming its outlook.", "Ford Motor Co. issued a press release confirming its outlook."),
])
def test_strip_style_leaks(raw, expected):
    from generate_data import strip_style_leaks
    assert strip_style_leaks(raw)[0] == expected


def test_primary_entity_added_when_missing():
    gen = {0: {**ITEM, "analysis": {**GOOD, "mentioned_entities": []}}}
    kept, _ = filter_examples([_spec(0)], gen, {0: GOOD})
    assert kept[0]["output"]["mentioned_entities"] == ["Nike Inc."]
    assert kept[0]["fixes"] == ["primary_entity_added_to_mentioned"]


def test_same_scenario_dropped_but_shared_templates_kept():
    mhi_a = {"company": "MHI", "event_type": "capital_return", "headline": "MHI announces $2 billion share buyback program",
             "snippet": "MHI said it will repurchase up to $2 billion of shares over twelve months to return excess cash."}
    mhi_b = {"company": "MHI", "event_type": "capital_return", "headline": "MHI announces $500 million share buyback program",
             "snippet": "MHI will repurchase up to $500 million of shares this year, funded from excess cash on hand."}
    adobe = {"company": "Adobe", "event_type": "guidance", "headline": "Adobe lifts third-quarter revenue guidance by 3%",
             "snippet": "Adobe raised its revenue outlook on strong demand for its creative cloud subscriptions."}
    tesla = {"company": "Tesla", "event_type": "guidance", "headline": "Tesla lifts third-quarter production guidance by 2%",
             "snippet": "Tesla now expects higher output after its Berlin plant ramped faster than planned."}
    kept, n = drop_near_duplicates([mhi_a, mhi_b, adobe, tesla])
    assert kept == [mhi_a, adobe, tesla] and n == 1


def test_split_is_80_10_10_overall_with_uneven_groups():
    sizes = [21, 19, 18, 18, 18, 18, 17, 16, 16, 15, 15, 12][:11]  # like the real data
    examples = [{"event_type": t, "id": f"{t}{i}"} for t, n in zip(EVENT_TYPES, sizes) for i in range(n)]
    splits = stratified_split(examples)
    n = len(examples)
    assert len(splits["test"]) == round(n * 0.1) and len(splits["val"]) == round(n * 0.1)
    assert {e["event_type"] for e in splits["test"]} == set(EVENT_TYPES)
    ids = [e["id"] for part in splits.values() for e in part]
    assert len(ids) == len(set(ids)) == n  # every example in exactly one split
