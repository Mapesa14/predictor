"""Name resolution and the promoted-team prior.

A loose fuzzy match is the worst failure this tool can have: it silently
predicts a different club and looks confident doing it. These pin that shut.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import markets, model  # noqa: E402
from predictor.engine import Predictor  # noqa: E402

# Real names that broke the loose matcher: every one of these is a distinct club.
LIGUE1 = ["Lens", "Lille", "Lyon", "Marseille", "Monaco", "Nantes", "Nice",
          "Paris SG", "Rennes", "Strasbourg", "Toulouse", "Auxerre"]
TURKEY = ["Pendikspor", "Bodrumspor", "Fenerbahce", "Galatasaray", "Besiktas",
          "Trabzonspor", "Alanyaspor", "Kasimpasa", "Samsunspor", "Eyupspor"]


def make_predictor(rows, weights=None):
    """A Predictor over a fixed match table, with no disk access."""
    p = Predictor.__new__(Predictor)
    p.root, p.xi, p.as_of = "", 0.0018, None
    p.goal_shrink, p.edge_scale = 0.30, 0.90
    p.use_ladder = False              # synthetic divisions are on no ladder
    p.weights = weights or {"goals": 1.0}
    p.market_weight = 0.0             # no prices in a synthetic league
    p._models = {}
    p.df = pd.DataFrame(rows)
    return p


def league(div, teams, seed=1):
    """Two round-robin seasons so the league can actually be fitted."""
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp("2024-08-01")
    for _ in range(2):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                rows.append({"Div": div, "Date": day, "HomeTeam": h, "AwayTeam": a,
                             "FTHG": int(rng.poisson(1.5)), "FTAG": int(rng.poisson(1.1)),
                             "HTHG": 0, "HTAG": 0, "Season": "2024/25"})
                day += pd.Timedelta(days=1)
    return rows


@pytest.fixture(scope="module")
def pred():
    return make_predictor(league("F1", LIGUE1) + league("T1", TURKEY, seed=2))


# ------------------------------------------------------------------ resolving
def test_exact_name_resolves(pred):
    assert pred.resolve("Lens", "F1")[0] == "Lens"


def test_case_and_space_insensitive(pred):
    assert pred.resolve("  lens ", "F1")[0] == "Lens"


def test_unique_prefix_resolves(pred):
    assert pred.resolve("Marse", "F1")[0] == "Marseille"


@pytest.mark.parametrize("typed,wrong", [
    ("Le Mans", "Lens"),            # promoted club, not Lens
    ("Amedspor", "Pendikspor"),     # promoted club, not Pendikspor
    ("Erzurumspor", "Bodrumspor"),  # promoted club, not Bodrumspor
])
def test_a_different_club_is_never_silently_substituted(pred, typed, wrong):
    div = "F1" if wrong == "Lens" else "T1"
    with pytest.raises(SystemExit):
        pred.resolve(typed, div)
    got, is_new = pred._resolve_or_new(typed, div)
    assert got == typed and is_new
    assert got != wrong


def test_strict_mode_never_fuzzy_matches(pred):
    with pytest.raises(SystemExit):
        pred.resolve("Lyonn", "F1", fuzzy=False)


def test_unknown_name_suggests_alternatives(pred):
    with pytest.raises(SystemExit) as e:
        pred.resolve("Zzzz Rovers", "F1")
    assert "Unknown team" in str(e.value)


def test_resolve_or_new_accepts_a_known_team(pred):
    got, is_new = pred._resolve_or_new("Nice", "F1")
    assert got == "Nice" and not is_new


# ------------------------------------------------------------ promoted prior
def test_an_unknown_team_gets_the_promoted_prior(pred):
    m = pred.models("F1")["FT"]
    lam, mu = m.rates("Lens", "Newly Promoted FC")
    exp_lam, exp_mu = _manual_rates(m, "Lens", None)
    assert lam == pytest.approx(exp_lam)
    assert mu == pytest.approx(exp_mu)


def _manual_rates(m, home, away):
    """Recompute rates by hand, with the away side as a newcomer."""
    ah = m.attack[home]
    dh = m.defence[home]
    aa = model.PROMOTED_ATTACK
    da = m.mean_defence + model.PROMOTED_DEFENCE
    log_lam = m.base + ah + da + m.home_adv
    log_mu = m.base + aa + dh
    level = 0.5 * (log_lam + log_mu)
    edge = 0.5 * (log_lam - log_mu)
    level = m.mean_log_rate + m.goal_shrink * (level - m.mean_log_rate)
    return float(np.exp(level + m.edge_scale * edge)), \
        float(np.exp(level - m.edge_scale * edge))


def test_a_newcomer_is_rated_worse_than_an_average_side(pred):
    m = pred.models("F1")["FT"]
    against_new = markets.result(m.score_matrix("Lens", "Newly Promoted FC"))
    # an average side sits at attack 0 and the league mean defence
    m.attack["Average FC"] = 0.0
    m.defence["Average FC"] = m.mean_defence
    against_avg = markets.result(m.score_matrix("Lens", "Average FC"))
    assert against_new["H"] > against_avg["H"]
    assert against_new["A"] < against_avg["A"]


def test_the_prior_is_the_measured_one():
    """Numbers come from 45 promoted team-seasons; see scratch/promoted.py."""
    assert model.PROMOTED_ATTACK == pytest.approx(-0.232)
    assert model.PROMOTED_DEFENCE == pytest.approx(0.150)
    assert np.exp(model.PROMOTED_ATTACK) == pytest.approx(0.79, abs=0.01)
    assert np.exp(model.PROMOTED_DEFENCE) == pytest.approx(1.16, abs=0.01)


def test_predict_flags_which_side_is_new(pred):
    s = pred.predict("Lens", "Le Mans", "F1", allow_new=True)
    assert s["away_new"] and not s["home_new"]
    assert s["away"] == "Le Mans"
    assert sum(s["result"].values()) == pytest.approx(1.0)


def test_predict_refuses_an_unknown_name_without_allow_new(pred):
    with pytest.raises(SystemExit):
        pred.predict("Lens", "Le Mans", "F1", allow_new=False)


def test_two_newcomers_in_one_fixture_are_symmetric(pred):
    s = pred.predict("New A", "New B", "F1", allow_new=True)
    assert s["home_new"] and s["away_new"]
    # identical ratings, so only home advantage separates them
    assert s["result"]["H"] > s["result"]["A"]
    assert s["exp_home"] > s["exp_away"]


# ---------------------------------------------------------------- aliases
@pytest.mark.parametrize("typed,expected", [
    ("Man Utd", "Man United"),
    ("man utd", "Man United"),
    ("Man U", "Man United"),
    ("Man United FC", "Man United"),
])
def test_common_abbreviations_resolve_deterministically(typed, expected):
    p = make_predictor(league("E0", ["Man United", "Man City", "Arsenal",
                                     "Chelsea", "Liverpool", "Everton"]))
    assert p.resolve(typed, "E0")[0] == expected


def test_aliases_work_with_fuzzy_disabled():
    """Abbreviations must not depend on fuzzy matching, which batch mode turns off."""
    p = make_predictor(league("E0", ["Man United", "Man City", "Arsenal",
                                     "Chelsea", "Liverpool", "Everton"]))
    assert p.resolve("Man Utd", "E0", fuzzy=False)[0] == "Man United"


def test_an_alias_never_reaches_a_different_club():
    p = make_predictor(league("F1", LIGUE1))
    with pytest.raises(SystemExit):
        p.resolve("Le Mans", "F1", fuzzy=False)
    with pytest.raises(SystemExit):
        p.resolve("Le Mans FC", "F1", fuzzy=False)
