"""The tip lists. Each rule here exists to stop a list flattering itself.

The measurement behind them is in predictor/tips.py: confidence filters for
accuracy, not profit; a shortlist is not a safe accumulator; and disagreeing
with the price is where the model is worst.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import tips  # noqa: E402

NOW = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)
LATER = "2026-09-13T18:00:00+03:00"


def row(home, away, p1, px, p2, market="same", div="E0", ko=LATER, started=None):
    m = {"div": div, "league": "League " + div, "date": ko, "home": home,
         "away": away, "p": {"1": p1, "X": px, "2": p2}, "score": "1-0"}
    if market == "same":
        m["market"] = {"1": p1, "X": px, "2": p2}
    else:
        m["market"] = market
    if started is not None:
        m["started"] = started
    return m


def strong(i, p=0.80, div=None):
    return row("H%d" % i, "A%d" % i, p, round((1 - p) * 0.6, 4),
               round((1 - p) * 0.4, 4), div=div or "D%d" % i)


# ------------------------------------------------------------- thresholds
def test_bankers_clear_the_bar_and_come_most_confident_first():
    rows = [strong(1, 0.78), strong(2, 0.86), strong(3, 0.70), strong(4, 0.81)]
    out = tips.build(rows, now=NOW)
    assert [c["p"] for c in out["bankers"]] == [0.86, 0.81, 0.78]
    assert all(c["p"] >= tips.BANKER_MIN for c in out["bankers"])


def test_the_short_list_is_never_padded():
    """Three qualifying fixtures means three bankers, not four."""
    rows = [strong(1, 0.80), strong(2, 0.79), strong(3, 0.77), strong(4, 0.70)]
    out = tips.build(rows, now=NOW)
    assert len(out["bankers"]) == 3
    assert any("not padded" in n for n in out["notes"])


def test_the_lists_stop_at_eight_and_twenty():
    rows = [strong(i, 0.80) for i in range(30)]
    out = tips.build(rows, now=NOW)
    assert len(out["bankers"]) == tips.BANKER_MAX
    assert len(out["long_list"]) == tips.LONG_MAX


def test_the_long_list_is_a_superset_at_the_lower_bar():
    rows = [strong(1, 0.82), strong(2, 0.68), strong(3, 0.60)]
    out = tips.build(rows, now=NOW)
    assert {c["home"] for c in out["bankers"]} <= {c["home"] for c in out["long_list"]}
    assert [c["home"] for c in out["long_list"]] == ["H1", "H2"]


def test_no_more_than_three_bankers_from_one_league():
    rows = [strong(i, 0.80 + i / 100, div="P1") for i in range(5)]
    out = tips.build(rows, now=NOW)
    assert len(out["bankers"]) == 3
    assert any("from one league" in n for n in out["notes"])
    assert len(out["long_list"]) == 5          # the cap is for the short list


# ------------------------------------------------------------ exclusions
def test_disagreeing_with_the_price_is_never_tipped():
    """121 such picks won 29.8% of the time."""
    against = row("Home", "Away", 0.78, 0.12, 0.10,
                  market={"1": 0.30, "X": 0.25, "2": 0.45})
    out = tips.build([against], now=NOW)
    assert out["bankers"] == [] and out["long_list"] == []
    assert [c["home"] for c in out["avoid"]] == ["Home"]


def test_unpriced_fixtures_get_their_own_section():
    """No price to agree with and no benchmark yet - Tanzania and CAF."""
    tz = row("Young Africans SC", "Geita Gold FC", 0.82, 0.11, 0.07,
             market=None, div="TZ1")
    out = tips.build([tz], now=NOW)
    assert out["bankers"] == [] and out["long_list"] == []
    assert [c["home"] for c in out["unpriced"]] == ["Young Africans SC"]


def test_nothing_already_under_way():
    rows = [strong(1, 0.85),
            row("Late", "Kick", 0.85, 0.09, 0.06, started=True),
            row("Past", "Time", 0.85, 0.09, 0.06, ko="2026-09-13T05:00:00+00:00")]
    out = tips.build(rows, now=NOW)
    assert [c["home"] for c in out["bankers"]] == ["H1"]
    assert out["excluded_started"] == 2


def test_a_draw_is_never_tipped():
    draw = row("Even", "Match", 0.10, 0.80, 0.10)
    out = tips.build([draw], now=NOW)
    assert out["bankers"] == [] and out["long_list"] == []


# -------------------------------------------------------------- numbers
def test_the_accumulator_is_the_product_not_a_promise():
    rows = [strong(1, 0.80), strong(2, 0.80), strong(3, 0.80), strong(4, 0.80)]
    acca = tips.build(rows, now=NOW)["accumulator"]
    assert acca["legs"] == 4
    assert acca["all_win"] == pytest.approx(0.80 ** 4)            # ~41%, not "sure"
    assert acca["all_win_double_chance"] == pytest.approx((0.80 + 0.12) ** 4)


def test_double_chance_adds_the_draw():
    c = tips.build([row("A", "B", 0.76, 0.14, 0.10)], now=NOW)["bankers"][0]
    assert c["p_double_chance"] == pytest.approx(0.90)
    assert c["side"] == "A" and c["pick"] == "1"


def test_an_away_favourite_names_the_away_side():
    c = tips.build([row("A", "B", 0.08, 0.14, 0.78)], now=NOW)["bankers"][0]
    assert (c["pick"], c["side"]) == ("2", "B")


def test_the_evidence_matches_the_thresholds_in_use():
    """If a threshold moves without re-measuring, the numbers on screen lie."""
    assert tips.EVIDENCE["bankers"]["threshold"] == tips.BANKER_MIN
    assert tips.EVIDENCE["long_list"]["threshold"] == tips.LONG_MIN


# --------------------------------------------------------------- record
def test_the_rule_is_scored_on_every_settled_prediction():
    settled = pd.DataFrame([
        # priced, agrees, 80% home -> banker; home won
        dict(pH=0.80, pD=0.12, pA=0.08, mk1=0.78, mkX=0.13, mk2=0.09, FTR="H"),
        # priced, agrees, 70% home -> long list only; draw
        dict(pH=0.70, pD=0.18, pA=0.12, mk1=0.66, mkX=0.20, mk2=0.14, FTR="D"),
        # priced, disagrees -> avoid; model pick (home) lost
        dict(pH=0.55, pD=0.25, pA=0.20, mk1=0.30, mkX=0.25, mk2=0.45, FTR="A"),
        # unpriced 75% home -> unpriced; home won
        dict(pH=0.75, pD=0.15, pA=0.10, mk1=None, mkX=None, mk2=None, FTR="H"),
        # not settled yet -> ignored
        dict(pH=0.90, pD=0.06, pA=0.04, mk1=0.9, mkX=0.06, mk2=0.04, FTR=None),
    ])
    r = tips.record_performance(settled)
    assert r["settled"] == 4
    assert (r["bankers"]["n"], r["bankers"]["hit"]) == (1, 1.0)
    assert (r["long_list"]["n"], r["long_list"]["hit"]) == (2, 0.5)
    assert (r["avoid"]["n"], r["avoid"]["hit"]) == (1, 0.0)
    assert (r["unpriced"]["n"], r["unpriced"]["hit"]) == (1, 1.0)
    assert r["bankers"]["enough"] is False


def test_an_empty_record_reports_nothing():
    r = tips.record_performance(pd.DataFrame())
    assert r["settled"] == 0 and r["bankers"]["hit"] is None
