"""Consistency checks: every market comes off one matrix, so nothing may clash."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import markets, model  # noqa: E402

RATES = [(1.9, 0.9), (0.8, 0.8), (2.6, 2.4), (0.4, 3.1), (1.3, 1.35)]
RHOS = [-0.12, 0.0, 0.06]


def mats(lam, mu, rho=-0.08):
    ft = model.score_matrix_from_rates(lam, mu, rho)
    f1 = model.score_matrix_from_rates(lam * 0.44, mu * 0.44, rho)
    f2 = model.score_matrix_from_rates(lam * 0.56, mu * 0.56, rho)
    return ft, f1, f2


@pytest.mark.parametrize("lam,mu", RATES)
@pytest.mark.parametrize("rho", RHOS)
def test_matrix_is_a_distribution(lam, mu, rho):
    m = model.score_matrix_from_rates(lam, mu, rho)
    assert m.min() >= 0
    assert m.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("lam,mu", RATES)
def test_result_and_double_chance_agree(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    r = markets.result(m)
    dc = markets.double_chance(m)
    assert sum(r.values()) == pytest.approx(1.0)
    assert dc["1X"] == pytest.approx(r["H"] + r["D"])
    assert dc["1X"] + dc["X2"] + dc["12"] == pytest.approx(2.0)


@pytest.mark.parametrize("lam,mu", RATES)
def test_totals_are_monotone(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    t = markets.totals(m)
    lines = sorted(t)
    overs = [t[ln]["over"] for ln in lines]
    assert overs == sorted(overs, reverse=True)
    for ln in lines:
        assert t[ln]["over"] + t[ln]["under"] == pytest.approx(1.0)


@pytest.mark.parametrize("lam,mu", RATES)
def test_over05_equals_not_nil_nil(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    assert markets.totals(m)[0.5]["over"] == pytest.approx(1.0 - m[0, 0])


@pytest.mark.parametrize("lam,mu", RATES)
def test_btts_matches_team_totals(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    tt = markets.team_totals(m)
    cs = markets.clean_sheet(m)
    # away failing to score is exactly the home clean sheet
    assert tt["away"][0.5]["under"] == pytest.approx(cs["home"])
    assert tt["home"][0.5]["under"] == pytest.approx(cs["away"])
    b = markets.btts(m)
    assert b["yes"] + b["no"] == pytest.approx(1.0)
    assert b["no"] == pytest.approx(cs["home"] + cs["away"] - m[0, 0])


@pytest.mark.parametrize("lam,mu", RATES)
def test_win_to_nil_is_a_subset_of_the_win(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    r = markets.result(m)
    w = markets.win_to_nil(m)
    assert w["home"] <= r["H"] + 1e-12
    assert w["away"] <= r["A"] + 1e-12


@pytest.mark.parametrize("lam,mu", RATES)
def test_asian_handicap_is_monotone_and_normalised(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    ah = markets.asian_handicap(m)
    lines = sorted(ah)
    homes = [ah[ln]["home"] for ln in lines]
    assert homes == sorted(homes)          # a friendlier line never hurts
    for ln in lines:
        assert ah[ln]["home"] + ah[ln]["away"] == pytest.approx(1.0)
    # the level line is draw-no-bet
    dnb = markets.draw_no_bet(m)
    assert ah[0.0]["home"] == pytest.approx(dnb["H"])


@pytest.mark.parametrize("lam,mu", RATES)
def test_european_handicap_sums_to_one(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    for ln, d in markets.european_handicap(m).items():
        assert sum(d.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("lam,mu", RATES)
def test_winning_margin_partitions_the_result(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    wm = markets.winning_margin(m)
    r = markets.result(m)
    home = sum(v for k, v in wm.items() if k.startswith("home"))
    away = sum(v for k, v in wm.items() if k.startswith("away"))
    assert home == pytest.approx(r["H"])
    assert away == pytest.approx(r["A"])
    assert sum(wm.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("lam,mu", RATES)
def test_correct_scores_are_ranked(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    ps = [p for _, _, p in markets.correct_scores(m, 12)]
    assert ps == sorted(ps, reverse=True)


@pytest.mark.parametrize("lam,mu", RATES)
def test_expected_goals_match_the_rates(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, 0.0)   # no rho, so exactly Poisson
    s = markets.summary(m)
    assert s["exp_home"] == pytest.approx(lam, abs=1e-3)
    assert s["exp_away"] == pytest.approx(mu, abs=1e-3)


@pytest.mark.parametrize("lam,mu", RATES)
def test_ht_ft_is_a_distribution_and_agrees_with_ht(lam, mu):
    ft, f1, f2 = mats(lam, mu)
    j = markets.ht_ft(f1, f2)
    assert sum(j.values()) == pytest.approx(1.0)
    ht = markets.result(f1)
    for side in "HDA":
        marg = sum(v for (h, _), v in j.items() if h == side)
        assert marg == pytest.approx(ht[side], abs=2e-3)


@pytest.mark.parametrize("lam,mu", RATES)
def test_combined_markets_sum_to_one(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    c = markets.combined(m)
    assert sum(v for k, v in c.items() if k.endswith("O2.5")) + \
           sum(v for k, v in c.items() if k.endswith("U2.5")) == pytest.approx(1.0)


@pytest.mark.parametrize("lam,mu", RATES)
def test_odd_even_and_half_most_goals(lam, mu):
    ft, f1, f2 = mats(lam, mu)
    oe = markets.odd_even(ft)
    assert oe["odd"] + oe["even"] == pytest.approx(1.0)
    hm = markets.half_with_most_goals(f1, f2)
    assert sum(hm.values()) == pytest.approx(1.0)


def test_summary_covers_every_market():
    ft, f1, f2 = mats(1.7, 1.1)
    s = markets.summary(ft, f1, f2)
    for key in ["result", "double_chance", "draw_no_bet", "totals", "team_totals",
                "btts", "clean_sheet", "win_to_nil", "odd_even", "correct_scores",
                "winning_margin", "asian_handicap", "european_handicap", "combined",
                "ht_result", "ht_totals", "ht_btts", "ht_ft", "half_most_goals"]:
        assert key in s, key


def test_stronger_team_gets_a_higher_win_probability():
    weak = markets.result(model.score_matrix_from_rates(1.2, 1.2, -0.08))
    strong = markets.result(model.score_matrix_from_rates(2.2, 0.8, -0.08))
    assert strong["H"] > weak["H"]
    assert strong["A"] < weak["A"]


def test_rho_lifts_the_draw_without_breaking_the_total():
    a = markets.result(model.score_matrix_from_rates(1.4, 1.3, 0.0))
    b = markets.result(model.score_matrix_from_rates(1.4, 1.3, -0.12))
    assert b["D"] > a["D"]
    assert sum(b.values()) == pytest.approx(1.0)


# ------------------------------------------------ the card's signature grid
@pytest.mark.parametrize("lam,mu", RATES)
def test_score_grid_agrees_with_the_matrix_it_came_from(lam, mu):
    """The grid and the correct-score list must never disagree about a score.

    The grid was renormalised over its 6x6 slice once, which inflated every
    cell by about 3% and had the heatmap and the correct-score list printing
    different numbers for the same scoreline.
    """
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    g = markets.score_grid(m, max_goals=5)
    for h in range(6):
        for a in range(6):
            assert g[h][a] == pytest.approx(m[h][a], abs=1e-6)
    top = markets.correct_scores(m, 1)[0]
    if top[0] <= 5 and top[1] <= 5:
        assert g[top[0]][top[1]] == pytest.approx(top[2], abs=1e-6)


@pytest.mark.parametrize("lam,mu", RATES)
def test_score_grid_remainder_completes_the_grid(lam, mu):
    m = model.score_matrix_from_rates(lam, mu, -0.08)
    g = markets.score_grid(m, max_goals=5)
    rest = markets.score_grid_remainder(m, max_goals=5)
    # the grid is rounded to 6dp per cell, so 36 cells carry a little slack
    total = sum(sum(r) for r in g) + rest
    assert total == pytest.approx(1.0, abs=1e-4)
    assert rest >= 0.0
