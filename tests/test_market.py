"""De-vigging, reading a price back into goal rates, and blending.

The load-bearing property here is separation: the published forecast may
contain the price, but the model-only view must survive untouched, or the
value comparison becomes circular and every "edge" it reports is invented.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import market, markets, model  # noqa: E402
from predictor.engine import Predictor  # noqa: E402

BOOKS = [
    [1.30, 6.00, 8.50],       # heavy favourite
    [2.42, 3.41, 2.65],       # close game
    [5.99, 4.11, 1.52],       # away favourite
    [2.00, 2.00, 2.00],       # flat, for symmetry
]


# ---------------------------------------------------------------- de-vigging
@pytest.mark.parametrize("odds", BOOKS)
@pytest.mark.parametrize("method", ["proportional", "shin"])
def test_devig_returns_a_distribution(odds, method):
    p = market.devig(odds, method)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()


@pytest.mark.parametrize("odds", BOOKS)
def test_devig_preserves_the_favourite(odds):
    p = market.devig(odds, "shin")
    assert np.argmax(p) == np.argmin(odds)


def test_devig_removes_the_margin():
    odds = [1.30, 6.00, 8.50]
    assert market.overround(odds) > 0.02
    for method in ("proportional", "shin"):
        assert market.devig(odds, method).sum() == pytest.approx(1.0)


def test_shin_corrects_the_favourite_longshot_bias():
    """Longshots are over-bet, so their naive implied probability overstates
    the truth. Shin lifts the favourite and shades the long prices."""
    for odds, fav, dog in ([1.30, 6.00, 8.50], 0, 2), ([5.99, 4.11, 1.52], 2, 0):
        prop = market.devig(odds, "proportional")
        shin = market.devig(odds, "shin")
        assert shin[fav] > prop[fav]
        assert shin[dog] < prop[dog]
        assert shin.sum() == pytest.approx(1.0)


def test_a_fair_book_is_left_alone():
    fair = [3.0, 3.0, 3.0]
    assert market.overround(fair) == pytest.approx(0.0)
    assert market.devig(fair, "shin") == pytest.approx([1 / 3] * 3)


def test_equal_prices_give_equal_probabilities():
    p = market.devig([2.0, 2.0, 2.0], "shin")
    assert p == pytest.approx([1 / 3] * 3)


# ------------------------------------------------------- price -> goal rates
@pytest.mark.parametrize("odds", BOOKS)
@pytest.mark.parametrize("rho", [0.0, -0.08])
def test_implied_rates_reproduce_the_prices(odds, rho):
    p = market.devig(odds, "shin")
    lam, mu = market.implied_rates(p[0], p[1], p[2], rho)
    got = markets.result(model.score_matrix_from_rates(lam, mu, rho))
    for key, want in zip("HDA", p):
        assert got[key] == pytest.approx(want, abs=0.02)


def test_implied_rates_respect_who_is_favourite():
    p = market.devig([1.30, 6.00, 8.50], "shin")
    lam, mu = market.implied_rates(p[0], p[1], p[2])
    assert lam > mu
    p = market.devig([5.99, 4.11, 1.52], "shin")
    lam, mu = market.implied_rates(p[0], p[1], p[2])
    assert lam < mu


def test_implied_rates_move_with_the_totals_price():
    """A shorter over price must imply more goals."""
    p = market.devig([2.42, 3.41, 2.65], "shin")
    low = market.implied_rates(p[0], p[1], p[2], -0.05, p_over25=0.35)
    high = market.implied_rates(p[0], p[1], p[2], -0.05, p_over25=0.65)
    assert sum(high) > sum(low)


def test_rates_from_row_reads_the_consensus_columns():
    row = pd.Series({"AvgH": 2.42, "AvgD": 3.41, "AvgA": 2.65,
                     "Avg>2.5": 1.90, "Avg<2.5": 1.90})
    lam, mu = market.rates_from_row(row)
    assert 0.5 < lam < 3.5 and 0.5 < mu < 3.5


def test_rates_from_row_falls_back_to_another_bookmaker():
    row = pd.Series({"AvgH": np.nan, "AvgD": np.nan, "AvgA": np.nan,
                     "B365H": 2.42, "B365D": 3.41, "B365A": 2.65})
    assert market.rates_from_row(row) is not None


def test_rates_from_row_returns_none_without_prices():
    assert market.rates_from_row(pd.Series({"HomeTeam": "X"})) is None
    assert market.rates_from_row(
        pd.Series({"AvgH": np.nan, "AvgD": 3.4, "AvgA": 2.6})) is None


def test_nonsense_prices_are_rejected():
    """Odds at or below 1.0 pay less than the stake and cannot be real."""
    assert market.rates_from_row(
        pd.Series({"AvgH": 0.5, "AvgD": 3.4, "AvgA": 2.6})) is None


# ------------------------------------------------------------------- blending
def test_blend_endpoints_are_the_two_inputs():
    m, k = (1.5, 1.1), (2.0, 0.8)
    assert market.blend_rates(m, k, 0.0) == pytest.approx(m)
    assert market.blend_rates(m, k, 1.0) == pytest.approx(k)


def test_blend_sits_between_and_moves_monotonically():
    m, k = (1.5, 1.1), (2.0, 0.8)
    prev = m[0]
    for w in (0.2, 0.4, 0.6, 0.8, 1.0):
        lam, mu = market.blend_rates(m, k, w)
        assert m[0] <= lam <= k[0]
        assert k[1] <= mu <= m[1]
        assert lam >= prev
        prev = lam


def test_blend_without_a_price_returns_the_model_untouched():
    m = (1.5, 1.1)
    assert market.blend_rates(m, None, 0.9) == pytest.approx(m)


def test_blend_weight_is_clamped():
    m, k = (1.5, 1.1), (2.0, 0.8)
    assert market.blend_rates(m, k, 5.0) == pytest.approx(k)
    assert market.blend_rates(m, k, -1.0) == pytest.approx(m)


# --------------------------------------------------- separation in the engine
def _predictor(rows, market_weight):
    p = Predictor.__new__(Predictor)
    p.root, p.xi, p.as_of = "", 0.0, None
    p.goal_shrink, p.edge_scale = 0.40, 1.10
    p.use_ladder, p.weights, p._models = False, {"goals": 1.0}, {}
    p.market_weight = market_weight
    p.df = pd.DataFrame(rows)
    return p


def _league():
    rng = np.random.default_rng(5)
    teams = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    rows, day = [], pd.Timestamp("2024-08-01")
    for _ in range(3):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                rows.append({"Div": "XX", "Date": day, "HomeTeam": h, "AwayTeam": a,
                             "FTHG": int(rng.poisson(1.5)), "FTAG": int(rng.poisson(1.1)),
                             "HTHG": 0, "HTAG": 0, "Season": "2024/25"})
                day += pd.Timedelta(days=1)
    return rows


ODDS = pd.Series({"AvgH": 5.99, "AvgD": 4.11, "AvgA": 1.52,
                  "Avg>2.5": 1.85, "Avg<2.5": 1.95})


def test_the_model_view_survives_the_blend():
    """Whatever the price says, the model's own numbers must be recoverable."""
    p = _predictor(_league(), market_weight=0.9)
    blended = p.predict("Alpha", "Bravo", "XX", odds=ODDS)
    pure = _predictor(_league(), market_weight=0.0).predict("Alpha", "Bravo", "XX")
    assert blended["model_result"] == pytest.approx(pure["result"])
    assert blended["model_exp_home"] == pytest.approx(pure["exp_home"], abs=1e-6)
    # and the published forecast really did move
    assert blended["result"]["A"] > blended["model_result"]["A"]
    assert blended["market_used"]


def test_no_odds_means_no_blend():
    p = _predictor(_league(), market_weight=0.9)
    s = p.predict("Alpha", "Bravo", "XX")
    assert not s["market_used"]
    assert s["market_weight"] == 0.0
    assert s["result"] == pytest.approx(s["model_result"])


def test_zero_weight_leaves_the_forecast_alone():
    p = _predictor(_league(), market_weight=0.0)
    s = p.predict("Alpha", "Bravo", "XX", odds=ODDS)
    assert s["result"] == pytest.approx(s["model_result"])


def test_the_blended_card_is_still_one_distribution():
    """Blending must not break the guarantee that markets cannot contradict."""
    p = _predictor(_league(), market_weight=0.9)
    s = p.predict("Alpha", "Bravo", "XX", odds=ODDS)
    assert sum(s["result"].values()) == pytest.approx(1.0)
    assert s["double_chance"]["1X"] == pytest.approx(
        s["result"]["H"] + s["result"]["D"])
    assert s["totals"][2.5]["over"] + s["totals"][2.5]["under"] == pytest.approx(1.0)
    assert s["asian_handicap"][0.0]["home"] == pytest.approx(s["draw_no_bet"]["H"])


def test_half_time_follows_the_blended_full_time():
    """The halves are nudged by the same factor, so they stay consistent."""
    p = _predictor(_league(), market_weight=0.9)
    s = p.predict("Alpha", "Bravo", "XX", odds=ODDS)
    pure = _predictor(_league(), market_weight=0.0).predict("Alpha", "Bravo", "XX")
    # the price makes the away side stronger, so its half-time chance must rise
    assert s["ht_result"]["A"] > pure["ht_result"]["A"]
    assert sum(s["ht_result"].values()) == pytest.approx(1.0)


def test_value_bets_never_use_a_market_blended_probability():
    """The regression this guards against silently zeroes out every edge."""
    from predictor import backtest
    bt = pd.DataFrame({
        "Date": [pd.Timestamp("2025-01-01")], "Div": ["XX"],
        "Home": ["Alpha"], "Away": ["Bravo"], "FTR": ["H"],
        "pH": [0.20], "pD": [0.30], "pA": [0.50],      # blended, agrees with price
        "mH": [0.60], "mD": [0.25], "mA": [0.15],      # model dissents
        "AvgH": [3.00], "AvgD": [3.40], "AvgA": [2.20],
    })
    v = backtest.value_bets(bt, edge=0.05)
    assert len(v) == 1
    assert v.iloc[0]["sel"] == "H"
    assert v.iloc[0]["p"] == pytest.approx(0.60)
