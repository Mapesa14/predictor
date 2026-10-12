"""The coupon: a week's clear picks, in one style or mixed.

What these defend: a list that is long because the week is busy, never because
the bar was lowered; one line per fixture, so a coupon cannot stack three bets
on the same match and call it a spread; and a mixed coupon that is actually
mixed, rather than forty copies of the most confident market on the board.
"""
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import markets, model, tips  # noqa: E402

NOW = datetime(2026, 10, 12, 6, 0, tzinfo=timezone.utc)


def row(home, away, lam, mu, div="E0", ko="2026-10-13T18:00:00+03:00"):
    s = markets.summary(model.score_matrix_from_rates(lam, mu, -0.05))
    r = s["result"]
    return {"div": div, "league": "League " + div, "date": ko,
            "home": home, "away": away,
            "p": {"1": r["H"], "X": r["D"], "2": r["A"]},
            "market": None, "markets": tips.selections(s),
            "o25": s["totals"][2.5]["over"], "btts": s["btts"]["yes"]}


def mixed_week():
    """A week with something for every market: thrashings, tight games, and
    the kind of open match where both teams score."""
    rows = []
    for i in range(6):                       # lopsided: wins, goals
        rows.append(row("Big%d" % i, "Small%d" % i, 2.8, 0.5, div="D%d" % i))
    for i in range(6):                       # tight and low scoring
        rows.append(row("Tight%d" % i, "Dour%d" % i, 0.8, 0.7, div="E%d" % i))
    for i in range(6):                       # open, both likely to score
        rows.append(row("Open%d" % i, "Loose%d" % i, 1.7, 1.5, div="F%d" % i))
    return rows


# ------------------------------------------------------------------ styles
def test_match_result_only_lists_nothing_but_match_results():
    c = tips.coupon(mixed_week(), style="normal", size=20, now=NOW)
    assert set(c["by_category"]) <= {"win", "double_chance"}
    assert c["style_label"] == "Match result only"


def test_mixed_spreads_across_markets():
    """Without the spread rule every line is the single most confident market
    on the board, and "mixed" means nothing."""
    c = tips.coupon(mixed_week(), style="mixed", size=12, now=NOW)
    assert len(c["by_category"]) >= 3, c["by_category"]
    assert max(c["by_category"].values()) <= max(1, round(12 * tips.CATEGORY_SHARE))


def test_an_unknown_style_falls_back_to_mixed():
    assert tips.coupon(mixed_week(), style="nonsense", now=NOW)["style"] == "mixed"


# ------------------------------------------------------------------- rules
def test_one_line_per_fixture():
    c = tips.coupon(mixed_week(), style="mixed", size=40, now=NOW)
    seen = [(p["home"], p["away"]) for p in c["picks"]]
    assert len(seen) == len(set(seen))


def test_a_quiet_week_gives_a_short_list_not_a_padded_one():
    c = tips.coupon([row("Even1", "Even2", 1.2, 1.25)], style="normal",
                    size=20, now=NOW)
    assert c["picks"] == []
    assert any("not padded" in n for n in c["notes"])


def test_nothing_below_its_own_bar_is_listed():
    c = tips.coupon(mixed_week(), style="mixed", size=50, now=NOW)
    for p in c["picks"]:
        assert p["p"] >= p["bar"]


def test_the_list_is_capped_at_the_size_asked_for():
    c = tips.coupon(mixed_week(), style="mixed", size=5, now=NOW)
    assert len(c["picks"]) == 5
    assert c["picks"] == sorted(c["picks"], key=lambda x: -x["p"])


def test_a_size_beyond_the_maximum_is_clamped():
    c = tips.coupon(mixed_week(), size=5000, now=NOW)
    assert c["size"] == tips.COUPON_MAX


def test_cup_ties_never_reach_the_coupon():
    rows = mixed_week()
    for m in rows:
        m["comp"] = "League Cup"
    assert tips.coupon(rows, size=20, now=NOW)["picks"] == []


def test_a_match_already_under_way_is_not_offered():
    past = row("Gone", "Already", 2.8, 0.5, ko="2026-10-12T05:00:00+03:00")
    assert tips.coupon([past], size=10, now=NOW)["picks"] == []


# ------------------------------------------------------------- the honesty
def test_the_headline_is_how_many_land_not_that_they_all_will():
    c = tips.coupon(mixed_week(), style="mixed", size=10, now=NOW)
    assert 0 < c["expected_correct"] <= len(c["picks"])
    assert c["expected_correct"] == pytest.approx(sum(p["p"] for p in c["picks"]), abs=0.01)
    note = " ".join(c["notes"])
    assert "not an accumulator" in note


def test_the_odds_of_the_whole_list_landing_are_stated_plainly():
    """However confident the list, the chance of every line landing is on the
    page - as a percentage while it is readable, and as "1 in N" once it is
    not. A reader should never have to work that out themselves."""
    c = tips.coupon(mixed_week(), style="mixed", size=12, now=NOW)
    note = " ".join(c["notes"])
    assert c["all_win"] < 1.0 and ("%" in note or "1 in" in note)
    assert tips._tiny(0.004).startswith("about 1 in")
    assert tips._tiny(0.2) == "20%"
    assert tips._tiny(0.0) == "effectively zero"


def test_every_pick_carries_what_its_market_actually_delivered():
    c = tips.coupon(mixed_week(), style="mixed", size=12, now=NOW)
    for p in c["picks"]:
        cat = next(x for x in tips.CATEGORIES if x["key"] == p["category"])
        leg = next(l for l in cat["legs"] if l["code"] == p["pick"])
        assert p["measured_hit"] == leg["hit"] and p["bar"] == leg["min"]
