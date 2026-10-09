"""Pick categories: goals lines, double chance, GG.

Each category is offered at a bar its own walk-forward measurement supports
(scratch/eval_markets.py). The failures these guard against are the ones that
make a tips product worthless: a category offered below the bar it was measured
at, a claim that does not match the measurement behind it, and a goals pick
settled as though it were a 1X2 pick.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import markets, model, tips, tipslog  # noqa: E402
from tests.test_tipslog import NOW, official  # noqa: E402

LATER = "2026-09-13T18:00:00+03:00"


def row(home, away, lam, mu, div="E0", ko=LATER):
    """A slate row built the way the service builds one."""
    s = markets.summary(model.score_matrix_from_rates(lam, mu, -0.05))
    r = s["result"]
    return {"div": div, "league": "League " + div, "date": ko,
            "home": home, "away": away,
            "p": {"1": r["H"], "X": r["D"], "2": r["A"]},
            "market": None, "markets": tips.selections(s),
            "o25": s["totals"][2.5]["over"], "btts": s["btts"]["yes"],
            "score": "2-0"}


def cat(out, key):
    return next(c for c in out["categories"] if c["key"] == key)


# ------------------------------------------------------------------- offers
def test_every_category_is_offered_at_the_bar_its_leg_was_measured_at():
    for c in tips.CATEGORIES:
        for leg in c["legs"]:
            assert tips.CATEGORY_MIN[leg["sel"]] == leg["min"], c["key"]
        # the headline is the legs weighted by how often each fires
        assert min(l["min"] for l in c["legs"]) == c["evidence"]["min"]
        assert (min(l["hit"] for l in c["legs"]) <= c["evidence"]["hit"]
                <= max(l["hit"] for l in c["legs"]))


def test_each_claim_matches_its_own_measurement():
    """The number on screen is the measured hit rate, not a rounded-up one."""
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scratch", "markets_eval.json")
    if not os.path.isfile(path):
        pytest.skip("measurement output not kept in this checkout")
    data = json.load(open(path, encoding="utf-8"))["selections"]
    for c in tips.CATEGORIES:
        for leg in c["legs"]:
            rows = data[leg["sel"]]["rows"]
            got = next(r for r in rows if abs(r["threshold"] - leg["min"]) < 1e-9)
            assert round(got["hit"], 3) == round(leg["hit"], 3), leg["sel"]
            assert round(got["model_said"], 3) == round(leg["model_said"], 3)
            assert round(got["n"]) == leg["n"], leg["sel"]


def test_a_weak_selection_is_not_offered_at_all():
    """No GG and over 3.5 were measured and left out: the model flatters itself
    on both, and one season of no-GG managed 46.7%."""
    offered = {leg["sel"] for c in tips.CATEGORIES for leg in c["legs"]}
    assert {n["label"] for n in tips.NOT_OFFERED} >= {"No both teams to score"}
    assert "btts_no" not in offered and "over35" not in offered
    assert "under25" not in offered


# -------------------------------------------------------------------- picks
def test_a_fixture_only_appears_where_it_clears_the_bar():
    out = tips.build([row("Man City", "Burnley", 2.6, 0.6)], now=NOW)
    assert [p["side"] for p in cat(out, "over05")["picks"]] == ["Over 0.5 goals"]
    assert [p["side"] for p in cat(out, "double_chance")["picks"]] == ["Man City or draw"]
    # a 1-1 kind of game clears nothing
    quiet = tips.build([row("Leeds", "Spurs", 0.9, 0.9)], now=NOW)
    assert all(not c["picks"] for c in quiet["categories"])


def test_picks_are_ordered_by_confidence_and_capped():
    rows = [row("H%d" % i, "A%d" % i, 2.4 + i / 50, 0.5, div="D%d" % i)
            for i in range(20)]
    picks = cat(tips.build(rows, now=NOW), "over15")["picks"]
    assert len(picks) == 12                                  # the category cap
    assert picks == sorted(picks, key=lambda c: -c["p"])


def test_a_started_fixture_is_never_offered():
    early = row("Man City", "Burnley", 2.6, 0.6, ko="2026-09-13T05:00:00+03:00")
    out = tips.build([early], now=NOW)
    assert all(not c["picks"] for c in out["categories"])


def test_cup_ties_stay_out_of_the_categories():
    m = row("Man City", "Burnley", 2.6, 0.6)
    m["comp"] = "League Cup"
    out = tips.build([m], now=NOW)
    assert all(not c["picks"] for c in out["categories"])


def test_a_goals_pick_is_not_given_a_price_it_does_not_have():
    """Only 1X2 carries a closing price in the feed."""
    m = row("Man City", "Burnley", 2.6, 0.6)
    m["market"] = {"1": 0.80, "X": 0.13, "2": 0.07}
    out = tips.build([m], now=NOW)
    assert cat(out, "over15")["picks"][0]["market_p"] is None
    assert cat(out, "win")["picks"][0]["market_p"] == 0.80


# --------------------------------------------------------------- settlement
@pytest.mark.parametrize("code,score,won", [
    ("over05", (0, 0), False), ("over05", (1, 0), True),
    ("over15", (1, 0), False), ("over15", (1, 1), True),
    ("over25", (2, 1), True), ("over25", (1, 1), False),
    ("1X", (1, 1), True), ("1X", (0, 1), False),
    ("X2", (1, 1), True), ("X2", (2, 1), False),
    ("btts_yes", (1, 1), True), ("btts_yes", (3, 0), False),
    ("1", (2, 0), True), ("2", (2, 0), False),
])
def test_each_pick_code_settles_on_the_score(code, score, won):
    assert tipslog._SETTLE[code](*score) is won


def test_a_frozen_category_pick_is_settled_from_the_result(tmp_path):
    rows = [row("Man City", "Burnley", 2.6, 0.6)]
    r = tipslog.freeze(rows, str(tmp_path), now=NOW)
    assert r["counts"]["over05"] == 1 and r["counts"]["double_chance"] == 1
    log = tipslog.load(str(tmp_path))
    s = tipslog.settle(log, official(("E0", "2026-09-13", "Man City", "Burnley", 0, 0)))
    got = {(x["list"], x["status"]) for _, x in s.iterrows()}
    assert ("over05", "lost") in got            # 0-0: no goal
    assert ("double_chance", "won") in got      # 0-0: City or draw
    assert tipslog.verify(str(tmp_path))["ok"]


def test_a_goals_pick_carries_no_double_chance(tmp_path):
    tipslog.freeze([row("Man City", "Burnley", 2.6, 0.6)], str(tmp_path), now=NOW)
    s = tipslog.settle(tipslog.load(str(tmp_path)),
                       official(("E0", "2026-09-13", "Man City", "Burnley", 2, 0)))
    goals = s[s["list"] == "over05"].iloc[0]
    win = s[s["list"] == "win"].iloc[0]
    assert goals["dc_won"] is None and win["dc_won"] is True


def test_every_category_has_a_promised_rate_to_be_judged_against():
    for c in tips.CATEGORIES:
        assert tipslog.PROMISED[c["key"]] == c["evidence"]["hit"]
        assert tipslog.LABELS[c["key"]] == c["label"]
