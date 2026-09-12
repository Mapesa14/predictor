"""Shot-blended rates and cross-division rating transfer.

Both were added because a measurement said they help; these pin down the
mechanics so a later change cannot quietly undo them.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import leagues, model  # noqa: E402
from predictor.engine import Predictor  # noqa: E402

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
TRUE_ATT = {"Alpha": .45, "Bravo": .25, "Charlie": 0., "Delta": -.1,
            "Echo": -.25, "Foxtrot": -.35}


def synthetic(div="XX", seasons=4, seed=11, sot_per_goal=3.0, start="2022-08-01"):
    """Seasons where shots and goals share one underlying team strength."""
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp(start)
    for _ in range(seasons):
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                lam = np.exp(.2 + TRUE_ATT[h] - TRUE_ATT[a] * .5 + .25)
                mu = np.exp(.2 + TRUE_ATT[a] - TRUE_ATT[h] * .5)
                hs, as_ = rng.poisson(lam * sot_per_goal), rng.poisson(mu * sot_per_goal)
                rows.append({"Div": div, "Date": day, "HomeTeam": h, "AwayTeam": a,
                             "FTHG": int(rng.binomial(hs, 1 / sot_per_goal)),
                             "FTAG": int(rng.binomial(as_, 1 / sot_per_goal)),
                             "HTHG": 0, "HTAG": 0,
                             "HST": float(hs), "AST": float(as_),
                             "HS": float(hs * 2), "AS": float(as_ * 2)})
                day += pd.Timedelta(days=2)
    df = pd.DataFrame(rows)
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    df["Season"] = df["Date"].map(lambda d: "%d/%02d" % (d.year, (d.year + 1) % 100)
                                  if d.month >= 7 else
                                  "%d/%02d" % (d.year - 1, d.year % 100))
    return df


@pytest.fixture(scope="module")
def sim():
    return synthetic()


# ------------------------------------------------------------------- targets
def test_shots_target_fits_and_reports_conversion(sim):
    m = model.fit(sim, "XX", "FT", xi=0.0, target="sot")
    assert m.target == "sot"
    # roughly one goal per three shots on target, by construction
    assert m.conversion == pytest.approx(1 / 3, abs=0.04)


def test_rho_is_pinned_off_for_shot_targets(sim):
    """The low-score correction is a goals phenomenon; shots never sit at 0-0."""
    assert model.fit(sim, "XX", "FT", xi=0.0, target="sot").rho == 0.0
    assert model.fit(sim, "XX", "FT", xi=0.0, target="shots").rho == 0.0


def test_shot_targets_reject_half_windows(sim):
    with pytest.raises(ValueError):
        model.fit(sim, "XX", "1H", xi=0.0, target="sot")


def test_unknown_target_is_rejected(sim):
    with pytest.raises(ValueError):
        model.fit(sim, "XX", "FT", target="corners")


def test_missing_shot_columns_raise(sim):
    bare = sim.drop(columns=["HST", "AST"])
    with pytest.raises(ValueError):
        model.fit(bare, "XX", "FT", target="sot")


# -------------------------------------------------------------------- blend
def test_blend_of_one_target_is_the_goals_model(sim):
    b = model.fit_blended(sim, "XX", xi=0.0, weights={"goals": 1.0})
    g = model.fit(sim, "XX", "FT", xi=0.0)
    assert isinstance(b, model.GoalModel)
    assert b.rates("Alpha", "Foxtrot") == pytest.approx(g.rates("Alpha", "Foxtrot"))


def test_blend_sits_between_its_two_components(sim):
    b = model.fit_blended(sim, "XX", xi=0.0, weights={"goals": 1.0, "sot": 1.0})
    assert isinstance(b, model.BlendedModel)
    g_lam, g_mu = b.goals.rates("Alpha", "Foxtrot")
    s = b.shots["sot"]
    s_lam, s_mu = s.rates("Alpha", "Foxtrot")
    s_lam, s_mu = s_lam * s.conversion, s_mu * s.conversion
    lam, mu = b.rates("Alpha", "Foxtrot")
    assert min(g_lam, s_lam) - 1e-9 <= lam <= max(g_lam, s_lam) + 1e-9
    assert min(g_mu, s_mu) - 1e-9 <= mu <= max(g_mu, s_mu) + 1e-9


def test_blend_weight_shifts_toward_the_heavier_side(sim):
    light = model.fit_blended(sim, "XX", xi=0.0, weights={"goals": 1.0, "sot": 0.1})
    heavy = model.fit_blended(sim, "XX", xi=0.0, weights={"goals": 0.1, "sot": 1.0})
    s = heavy.shots["sot"]
    pure_sot = s.rates("Alpha", "Foxtrot")[0] * s.conversion
    assert abs(heavy.rates("Alpha", "Foxtrot")[0] - pure_sot) < \
           abs(light.rates("Alpha", "Foxtrot")[0] - pure_sot)


def test_blended_model_exposes_the_goal_model_surface(sim):
    b = model.fit_blended(sim, "XX", xi=0.0, weights={"goals": 1.0, "sot": 1.0})
    for attr in ("div", "window", "teams", "attack", "defence", "home_adv",
                 "rho", "n_matches", "played", "mean_defence", "carried"):
        getattr(b, attr)
    assert b.rho == b.goals.rho
    m = b.score_matrix("Alpha", "Bravo")
    assert m.sum() == pytest.approx(1.0)


def test_blend_falls_back_when_shot_data_is_absent(sim):
    bare = sim.drop(columns=["HST", "AST", "HS", "AS"])
    b = model.fit_blended(bare, "XX", xi=0.0, weights={"goals": 1.0, "sot": 1.0})
    assert isinstance(b, model.GoalModel)


# --------------------------------------------------------------- the ladder
def test_rating_shift_is_signed_and_additive():
    up = leagues.rating_shift("E1", "E0")
    down = leagues.rating_shift("E0", "E1")
    assert up[0] == pytest.approx(-down[0])
    assert up[1] == pytest.approx(-down[1])
    # moving up costs attack and worsens defence
    assert up[0] < 0 and up[1] > 0
    two = leagues.rating_shift("E2", "E0")
    assert two[0] == pytest.approx(up[0] + leagues.rating_shift("E2", "E1")[0])


def test_rating_shift_is_zero_within_a_division():
    assert leagues.rating_shift("E0", "E0") == (0.0, 0.0)


def test_rating_shift_refuses_unlinked_divisions():
    assert leagues.rating_shift("E1", "SP1") is None
    assert leagues.rating_shift("D1", "E0") is None
    assert leagues.ladder_of("SP1") is None


def test_carried_ratings_do_not_pollute_the_league_table():
    """A club carried up must not appear as though it played in the division."""
    df = pd.concat([synthetic("E0", seasons=2, seed=1),
                    synthetic("E1", seasons=2, seed=2, start="2022-08-01")],
                   ignore_index=True)
    # give E1 a club that E0 has never seen
    extra = synthetic("E1", seasons=2, seed=3).copy()
    extra["HomeTeam"] = extra["HomeTeam"].replace({"Alpha": "Newtown"})
    extra["AwayTeam"] = extra["AwayTeam"].replace({"Alpha": "Newtown"})
    df = pd.concat([df, extra], ignore_index=True)

    p = Predictor.__new__(Predictor)
    p.root, p.xi, p.as_of = "", 0.0, None
    p.goal_shrink, p.edge_scale = 0.30, 0.90
    p.use_ladder, p.weights, p._models = True, {"goals": 1.0}, {}
    p.market_weight = 0.0
    p.df = df
    m = p.models("E0")["FT"]

    assert "Newtown" not in m.teams
    assert "Newtown" not in m.attack
    assert not m.knows("Newtown")
    assert "Newtown" in m.carried_attack
    assert m.rating_source("Newtown") == "E1"
    assert m.rating_source("Alpha") == "E0"
    listed = [t for t, *_ in model.strength_table(m)]
    assert "Newtown" not in listed


def test_a_carried_club_is_rated_below_its_lower_division_self():
    """The same club must look weaker one division up."""
    df = pd.concat([synthetic("E0", seasons=2, seed=1),
                    synthetic("E1", seasons=2, seed=2)], ignore_index=True)
    p = Predictor.__new__(Predictor)
    p.root, p.xi, p.as_of = "", 0.0, None
    p.goal_shrink, p.edge_scale = 0.30, 0.90
    p.use_ladder, p.weights, p._models = True, {"goals": 1.0}, {}
    p.market_weight = 0.0
    p.df = df
    e1 = p.models("E1")["FT"]
    e0 = p.models("E0")["FT"]
    for t in ("Bravo", "Delta"):
        if t in e0.carried_attack:
            assert e0.carried_attack[t] < e1.attack[t]


def test_ladder_can_be_switched_off():
    df = pd.concat([synthetic("E0", seasons=2, seed=1),
                    synthetic("E1", seasons=2, seed=2)], ignore_index=True)
    p = Predictor.__new__(Predictor)
    p.root, p.xi, p.as_of = "", 0.0, None
    p.goal_shrink, p.edge_scale = 0.30, 0.90
    p.use_ladder, p.weights, p._models = False, {"goals": 1.0}, {}
    p.market_weight = 0.0
    p.df = df
    assert p.models("E0")["FT"].carried_attack == {}


def test_rating_source_falls_back_to_the_prior(sim):
    m = model.fit(sim, "XX", "FT", xi=0.0)
    assert m.rating_source("Nobody FC") == "prior"
    a, d = m._team("Nobody FC")
    assert a == model.PROMOTED_ATTACK
    assert d == m.mean_defence + model.PROMOTED_DEFENCE


def test_a_carried_rating_beats_the_prior_as_the_fallback(sim):
    """Carried ratings take precedence over the flat prior, not the reverse."""
    m = model.fit(sim, "XX", "FT", xi=0.0)
    m.carried_attack["Newtown"] = 0.4
    m.carried_defence["Newtown"] = m.mean_defence - 0.3
    m.carried["Newtown"] = "YY"
    assert m._team("Newtown") == (0.4, m.mean_defence - 0.3)
    assert m.rating_source("Newtown") == "YY"
