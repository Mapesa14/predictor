"""Model and loader behaviour, checked on synthetic seasons with known truth."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import backtest, fixtures, loader, model  # noqa: E402

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
TRUE_ATT = {"Alpha": 0.45, "Bravo": 0.25, "Charlie": 0.0,
            "Delta": -0.1, "Echo": -0.25, "Foxtrot": -0.35}
TRUE_DEF = {"Alpha": -0.35, "Bravo": -0.15, "Charlie": 0.0,
            "Delta": 0.1, "Echo": 0.2, "Foxtrot": 0.4}
HOME_ADV, BASE = 0.26, 0.2


def synthetic(seasons=6, seed=7):
    """Simulate whole double round-robin seasons from known ratings."""
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp("2020-08-01")
    for _ in range(seasons):
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                lam = np.exp(BASE + TRUE_ATT[h] + TRUE_DEF[a] + HOME_ADV)
                mu = np.exp(BASE + TRUE_ATT[a] + TRUE_DEF[h])
                fh, fa = rng.poisson(lam), rng.poisson(mu)
                rows.append({"Div": "XX", "Date": day, "HomeTeam": h, "AwayTeam": a,
                             "FTHG": fh, "FTAG": fa,
                             "HTHG": rng.binomial(fh, 0.44),
                             "HTAG": rng.binomial(fa, 0.44)})
                day += pd.Timedelta(days=2)
    df = pd.DataFrame(rows)
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    df["Season"] = df["Date"].map(loader.season_of)
    df["TotalGoals"] = df.FTHG + df.FTAG
    return df


@pytest.fixture(scope="module")
def sim():
    return synthetic()


def test_recovers_home_advantage(sim):
    m = model.fit(sim, "XX", "FT", xi=0.0)
    assert m.home_adv == pytest.approx(HOME_ADV, abs=0.09)


def test_recovers_the_attack_ordering(sim):
    m = model.fit(sim, "XX", "FT", xi=0.0)
    order = [t for t, *_ in sorted(m.attack.items(), key=lambda kv: -kv[1])]
    truth = [t for t, _ in sorted(TRUE_ATT.items(), key=lambda kv: -kv[1])]
    # exact order can swap adjacent pairs on finite data; correlation must be high
    ranks_m = {t: i for i, t in enumerate(order)}
    ranks_t = {t: i for i, t in enumerate(truth)}
    x = np.array([ranks_m[t] for t in TEAMS], float)
    y = np.array([ranks_t[t] for t in TEAMS], float)
    assert np.corrcoef(x, y)[0, 1] > 0.85


def test_recovers_the_goal_rates(sim):
    m = model.fit(sim, "XX", "FT", xi=0.0)
    lam, mu = m.rates("Alpha", "Foxtrot")
    true_lam = np.exp(BASE + TRUE_ATT["Alpha"] + TRUE_DEF["Foxtrot"] + HOME_ADV)
    true_mu = np.exp(BASE + TRUE_ATT["Foxtrot"] + TRUE_DEF["Alpha"])
    assert lam == pytest.approx(true_lam, rel=0.20)
    assert mu == pytest.approx(true_mu, rel=0.30)


def test_attack_ratings_sum_to_zero(sim):
    m = model.fit(sim, "XX", "FT", xi=0.0)
    assert sum(m.attack.values()) == pytest.approx(0.0, abs=1e-6)


def test_analytic_gradient_matches_finite_differences(sim):
    """The hand-derived gradient is the reason fitting is fast - verify it."""
    from scipy.optimize import check_grad
    import predictor.model as M

    captured = {}
    real_min = M.minimize

    def spy(fun, x0, jac=None, **kw):
        captured["fun"], captured["jac"], captured["x0"] = fun, jac, x0
        return real_min(fun, x0, jac=jac, **kw)

    M.minimize = spy
    try:
        M.fit(sim, "XX", "FT", xi=0.0018)
    finally:
        M.minimize = real_min
    rng = np.random.default_rng(0)
    x = captured["x0"] + rng.normal(0, 0.15, size=captured["x0"].shape)
    err = check_grad(captured["fun"], captured["jac"], x)
    scale = np.linalg.norm(captured["jac"](x))
    assert err / max(scale, 1.0) < 1e-4


def test_time_decay_follows_recent_form():
    """A team that collapses late must be rated below its long-run average."""
    rng = np.random.default_rng(3)
    rows, day = [], pd.Timestamp("2021-08-01")
    for era, (att, n) in enumerate([(0.6, 150), (-0.6, 60)]):
        for _ in range(n):
            h, a = rng.choice(TEAMS[1:], 2, replace=False)
            for home, away, ha in ((("Alpha"), a, att), (h, "Alpha", att)):
                lam = np.exp(BASE + (ha if home == "Alpha" else 0.0) + HOME_ADV)
                mu = np.exp(BASE + (ha if away == "Alpha" else 0.0))
                fh, fa = rng.poisson(lam), rng.poisson(mu)
                rows.append({"Div": "XX", "Date": day, "HomeTeam": home,
                             "AwayTeam": away, "FTHG": fh, "FTAG": fa,
                             "HTHG": 0, "HTAG": 0})
                day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows)
    df["Season"] = df["Date"].map(loader.season_of)
    decayed = model.fit(df, "XX", "FT", xi=0.006).attack["Alpha"]
    flat = model.fit(df, "XX", "FT", xi=0.0).attack["Alpha"]
    assert decayed < flat


def test_goal_shrink_pulls_extreme_fixtures_toward_the_mean(sim):
    hard = model.fit(sim, "XX", "FT", xi=0.0, goal_shrink=0.5)
    raw = model.fit(sim, "XX", "FT", xi=0.0, goal_shrink=1.0)
    lam_h, mu_h = hard.rates("Alpha", "Foxtrot")
    lam_r, mu_r = raw.rates("Alpha", "Foxtrot")
    assert abs(np.log(lam_h * mu_h) / 2 - hard.mean_log_rate) < \
           abs(np.log(lam_r * mu_r) / 2 - raw.mean_log_rate)


def test_edge_scale_flattens_the_result(sim):
    m = model.fit(sim, "XX", "FT", xi=0.0, edge_scale=0.5)
    raw = model.fit(sim, "XX", "FT", xi=0.0, edge_scale=1.0)
    assert (m.rates("Alpha", "Foxtrot")[0] / m.rates("Alpha", "Foxtrot")[1]) < \
           (raw.rates("Alpha", "Foxtrot")[0] / raw.rates("Alpha", "Foxtrot")[1])


def test_neutral_venue_removes_home_advantage(sim):
    """Home advantage is a multiplier on the home rate only, as Dixon-Coles has it."""
    m = model.fit(sim, "XX", "FT", xi=0.0)
    lam, mu = m.rates("Alpha", "Bravo")
    nlam, nmu = m.rates("Alpha", "Bravo", neutral=True)
    assert nlam == pytest.approx(lam * np.exp(-m.home_adv))
    assert nmu == pytest.approx(mu)
    assert nlam < lam


def test_walk_forward_never_sees_the_future(sim):
    """The as-of cut is what makes the backtest honest, so pin it down."""
    cut = sim["Date"].iloc[100]
    m = model.fit(sim, "XX", "FT", as_of=cut)
    assert m.n_matches == int((sim["Date"] < cut).sum())


def test_rps_is_zero_for_a_perfect_call_and_worst_for_the_far_end():
    assert backtest.rps(np.array([1.0, 0.0, 0.0]), 0) == pytest.approx(0.0)
    assert backtest.rps(np.array([0.0, 0.0, 1.0]), 0) == pytest.approx(1.0)
    mid = backtest.rps(np.array([1 / 3, 1 / 3, 1 / 3]), 1)
    assert 0 < mid < 1


def test_season_labels_split_in_july():
    assert loader.season_of(pd.Timestamp("2025-08-15")) == "2025/26"
    assert loader.season_of(pd.Timestamp("2026-05-20")) == "2025/26"
    assert loader.season_of(pd.Timestamp("2026-07-02")) == "2026/27"


def test_remaining_fixtures_are_the_unplayed_pairings(sim):
    """Drop one fixture from the last season; it must reappear as outstanding."""
    last = sim["Season"].max()
    mask = sim["Season"] == last
    dropped = sim[mask].iloc[3]
    trimmed = sim.drop(index=sim[mask].index[3])
    rem = fixtures.remaining(trimmed, "XX")
    pairs = set(zip(rem["HomeTeam"], rem["AwayTeam"]))
    assert (dropped["HomeTeam"], dropped["AwayTeam"]) in pairs


def test_matchday_ordering_never_repeats_a_team_in_a_round():
    pairs = [("A", "B"), ("A", "C"), ("B", "C"), ("D", "E"), ("A", "D")]
    ordered = fixtures._matchdays(pairs)
    assert sorted(ordered) == sorted(pairs)
    seen = set()
    for h, a in ordered:
        if h in seen or a in seen:
            break
        seen.update((h, a))
    else:
        pytest.fail("expected the first round to end")
    assert ordered[0] == ("A", "B")
