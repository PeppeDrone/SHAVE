"""SHAVE target rule: Wilson 95% lower-bound ranking and conversion of a removal option to insertion."""
import json
from pathlib import Path

import numpy as np
import pytest

from interp_pilot import civil as C
from interp_pilot.shave import select_target, target_str, wilson_lb


def row(value, p0, n0, p1, n1, count=10):
    return {"value": value, "count": count, "override_to0": p0, "override_to1": p1, "n_to0": n0, "n_to1": n1}


SCAN44 = {"cc_comment": [row("x", 0.5, 150, 0.5, 150)],
          "cc_w:gay": [row("present", 0.0, 150, 0.30, 150), row("absent", 1.0, 3, 0.0, 3)],
          "cc_w:muslims": [row("present", 0.0, 150, 149 / 150, 150), row("absent", 0.5, 6, 0.0, 6)],
          "cc_w:people": [row("present", 0.02, 150, 0.05, 150), row("absent", 0.0, 140, 0.0, 140)]}
FIELDS44 = list(SCAN44)


def test_wilson_lb_values():
    assert wilson_lb(149 / 150, 150) > wilson_lb(1.0, 3)
    assert abs(wilson_lb(1.0, 3) - 0.4385) < 1e-3 and abs(wilson_lb(149 / 150, 150) - 0.9632) < 1e-3
    assert wilson_lb(0.0, 0) == 0.0 and wilson_lb(0.0, 50) == 0.0
    assert 0 < wilson_lb(0.5, 10) < 0.5


def test_wilson_ranking_prefers_149_of_150_over_3_of_3():
    new = select_target(SCAN44, FIELDS44, "r12")
    assert target_str(new) == "cc_w:muslims=present->c1"
    assert new["choice"]["converted_from"] is None and new["field_ranking"][0] == "cc_w:muslims"


def test_absent_target_converted_to_matching_present():
    scan = {"cc_w:muslims": [row("present", 0.0, 150, 0.70, 150), row("absent", 1.0, 40, 0.0, 40)],
            "cc_w:gay": [row("present", 0.0, 150, 0.2, 150), row("absent", 0.5, 4, 0.0, 4)]}
    new = select_target(scan, list(scan), "r12")
    assert new["fstar"] == "cc_w:muslims" and new["Vstar"] == ["present"] and new["cstar"] == 1
    cf = new["choice"]["converted_from"]
    assert cf["value"] == "absent" and cf["class"] == 0 and cf["n"] == 40
    assert new["choice"]["evidence"] == {"value": "present", "class": 1, "rate": 0.70, "n": 150, "wilson_lb": wilson_lb(0.70, 150)}
    assert select_target(SCAN44, FIELDS44, "r12")["Vstar"] == ["present"]


def test_absent_target_without_present_option_is_skipped():
    scan = {"cc_w:aaa": [row("absent", 1.0, 60, 0.0, 60)],
            "cc_w:bbb": [row("present", 0.0, 150, 0.0, 0), row("absent", 0.98, 50, 0.0, 50)],
            "cc_w:ccc": [row("present", 0.1, 150, 0.6, 150), row("absent", 0.2, 30, 0.0, 30)]}
    new = select_target(scan, list(scan), "r12")
    assert [x["field"] for x in new["skipped"]] == ["cc_w:aaa", "cc_w:bbb"]
    assert target_str(new) == "cc_w:ccc=present->c1"


def test_no_word_field_falls_back_to_coarse_field():
    scan = {"cc_comment": [row("x", 0.95, 100, 0.1, 100)]}
    d = select_target(scan, list(scan), "r12")
    assert d["fstar"] == "cc_comment" and d["choice"] is None and not C.is_word_field(d["fstar"])


def test_only_paper_rule_is_supported():
    with pytest.raises(ValueError):
        select_target(SCAN44, FIELDS44, "r10")
    with pytest.raises(ValueError):
        select_target(SCAN44, FIELDS44, "r12p")


def test_present_target_texts_have_nonzero_span_positions():
    texts = [C.make_text("these people are awful."), C.make_text("I like the new park!"), C.make_text("word " * 300),
             C.make_text("Muslims and muslims again")]
    for t in texts:
        t2 = C.replace_value(t, "cc_w:muslims", "present")
        assert C.field_value(t2, "cc_w:muslims") == "present"
        spans = C.all_word_char_spans(t2, "cc_w:muslims")
        assert spans and spans[0][0] == C.VAL_OFFSET
        t3 = C.replace_value(t2, "cc_w:muslims", "absent")
        assert C.all_word_char_spans(t3, "cc_w:muslims") == []
