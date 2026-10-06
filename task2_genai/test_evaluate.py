"""Offline tests for Task 2C scoring (no GPU, no network)."""
import csv
import json

import pytest

from evaluate import (hallucination_rate, lenient_json, paired_wins, score_item, strict_parse, summarise,
                      write_review_sheet)
from news_schemas import JudgeScore

NEWS = ("Headline: Nike Inc. cuts full-year revenue outlook as China demand slows\n"
        "Snippet: Nike Inc. lowered its fiscal-year revenue growth forecast to 2% from 5%, citing weaker demand.")
REF = {"event_type": "guidance", "primary_entity": "Nike Inc.", "mentioned_entities": ["Nike Inc."],
       "impact_direction": "negative", "magnitude": "high",
       "rationale": "Nike cut its growth forecast from 5% to 2%. A guidance cut resets expectations for the year."}
GOOD_RAW = json.dumps(REF)


def test_strict_vs_lenient_parsing():
    wrapped = "Here is the analysis:\n```json\n" + GOOD_RAW + "\n```"
    assert strict_parse(GOOD_RAW) is not None and strict_parse(wrapped) is None
    assert lenient_json(wrapped)["event_type"] == "guidance"
    assert lenient_json("no json here") is None and lenient_json("{broken") is None


def test_score_item_separates_format_from_labels():
    wrapped_extra_key = "Sure!\n" + json.dumps({**REF, "event_type": "Guidance", "confidence": 0.9})
    row = score_item(wrapped_extra_key, REF, NEWS)
    assert row["schema_valid"] is False                         # prose + extra key: format fails...
    assert row["event_type_correct"] and row["magnitude_correct"]  # ...but labels still count
    assert row["grounded"] is False                              # not schema-valid -> not counted as grounded


def test_score_item_flags_invented_number():
    invented = json.dumps({**REF, "rationale": "Nike cut guidance to 2%. Shares fell 9% after hours."})
    row = score_item(invented, REF, NEWS)
    assert row["schema_valid"] and not row["grounded"] and any("9" in i for i in row["grounding_issues"])


def test_summary_and_paired_wins():
    rows = [score_item(GOOD_RAW, REF, NEWS), score_item("garbage", REF, NEWS)]
    j = JudgeScore(label_accuracy=5, grounding=5, rationale_quality=4, format_compliance=5, verdict="correct",
                   justification="ok")
    s = summarise(rows, [1.0, 0.0], [1.0, 0.0], None, [j, None])
    assert s["schema_valid_%"] == 50.0 and s["event_type_acc_%"] == 50.0 and s["judge_failures"] == 1
    assert s["judge_correct_%"] == 100.0
    assert paired_wins([0.2, 0.5, 0.4], [0.3, 0.5, 0.1]) == {"b_better": 1, "tie": 1, "a_better": 1}


def test_review_sheet_round_trip(tmp_path):
    path = tmp_path / "review.csv"
    rows = [score_item(GOOD_RAW, REF, NEWS)] * 3
    write_review_sheet(path, [NEWS] * 3, [REF] * 3, [GOOD_RAW] * 3, rows, [None] * 3)
    assert hallucination_rate(path) is None  # nothing labelled yet

    data = list(csv.DictReader(path.open()))
    for r, label in zip(data, ["correct", "hallucinated", "partially_correct"]):
        r["manual_label"] = label
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    assert hallucination_rate(path) == {"reviewed": 3, "hallucination_rate_%": 33.3,
                                        "counts": {"correct": 1, "partially_correct": 1, "incorrect": 0, "hallucinated": 1}}


def test_bad_manual_label_rejected(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("id,manual_label\n0,maybe\n")
    with pytest.raises(ValueError):
        hallucination_rate(path)
