"""Pin the measured constants from BUILD-PROMPT.md §8 as regression tests.

Fast tests run always and need no data. The `slow` tests reproduce the
walk-forward and bridge numbers the whole product is built on and need the
soccer-data folder; they run only when FOOTBALL_DATA_ACCEPTANCE=1 is set.

    pytest tests -q                                  # fast suite
    $env:FOOTBALL_DATA_ACCEPTANCE=1; pytest -m slow  # the §8 acceptance run
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from predictor import backtest, engine, loader, markets, model
from predictor.backtest import prior_scan, score_blends

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"D:\Downloads July 2026\SoccerData"
SLOW_OK = (os.environ.get("FOOTBALL_DATA_ACCEPTANCE") == "1"
           and os.path.isdir(DATA))

skip_slow = pytest.mark.skipif(not SLOW_OK, reason=(
    "needs %s and FOOTBALL_DATA_ACCEPTANCE=1" % DATA))


# ------------------------------------------------------------- fast: engine

def test_xi_default_matches_engine():
    """One measured decay, everywhere. This default drifted to 0.0019 once."""
    import inspect

    def xi_default(fn):
        return inspect.signature(fn).parameters["xi"].default

    want = engine.DEFAULT_XI
    assert want == 0.0018
    assert xi_default(model.fit) == want
    assert xi_default(model.fit_all) == want
    assert xi_default(model.fit_blended) == want
    assert xi_default(backtest.walk_forward) == want


def test_shrink_defaults_match_measured_constants():
    """The per-family constants of §1.3, as packaged defaults."""
    assert engine.DEFAULT_GOAL_SHRINK == 0.40
    assert engine.DEFAULT_EDGE_SCALE == 1.10


def test_bridge_files_hold_the_measured_offsets():
    """The §1.5 tables, verbatim from bridge.json / bridge_caf.json."""
    b = engine.load_bridge()
    assert b is not None
    for code, want in [("E0", 0.454), ("F1", 0.382), ("D1", 0.287),
                       ("SP1", 0.243), ("I1", 0.212), ("N1", -0.024),
                       ("NOR1", -0.033), ("B1", -0.042), ("P1", -0.114),
                       ("AUT1", -0.185), ("G1", -0.191), ("SC0", -0.202),
                       ("T1", -0.213), ("UKR1", -0.258), ("CZE1", -0.317)]:
        assert b["offsets"][code] == pytest.approx(want, abs=0.005)
    assert b["home_adv"] == pytest.approx(0.297, abs=0.003)
    assert b["n"] == 439
    assert b["goal_shrink"] == 1.0
    assert b["edge_scale"] == 1.1

    c = engine.load_caf_bridge()
    assert c is not None
    for code, want in [("F-TUN", 0.497), ("F-COD", 0.427), ("EG1", 0.340),
                       ("DZ1", 0.198), ("MA1", 0.160), ("TZ1", 0.119),
                       ("ZA1", 0.110), ("F-MWI", -0.018), ("F-BOT", -0.273)]:
        assert c["offsets"][code] == pytest.approx(want, abs=0.005)
    assert c["home_adv"] == pytest.approx(0.585, abs=0.003)
    assert c["beta"] == 1.0
    assert c["n"] == 199
    assert c["goal_shrink"] == 1.0
    assert c["edge_scale"] == 1.1


def test_caf_tie_counts_that_informed_the_offsets_are_recorded():
    """§1.5: the UI must show how many CAF ties priced each federation."""
    c = engine.load_caf_bridge()
    assert c["matches"]["TZ1"] == 18
    assert c["matches"]["F-TUN"] == 28
    assert c["matches"]["F-COD"] == 27
    for code in c["offsets"]:
        if code.startswith("F-"):
            assert code in c["matches"]


# ------------------------------------------------------------- fast: ties

def _square(n):
    m = np.random.default_rng(7).random((n, n))
    return m / m.sum()


def test_tie_outcome_partitions_the_matrix():
    """win + level + lose must equal the whole matrix, and be non-negative."""
    ft = _square(13)
    r = engine.tie_outcome(ft, 0, 0)
    assert r["win"] + r["level"] + r["lose"] == pytest.approx(1.0)
    assert all(v >= 0.0 for v in r.values())


def test_tie_outcome_level_is_reported_not_resolved():
    """Level on aggregate goes to extra time; no away-goals rule is asserted."""
    ft = _square(13)
    r = engine.tie_outcome(ft, 0, 0)
    # Level must be exactly the h==a slice of the aggregate score matrix.
    assert r["level"] == pytest.approx(ft.diagonal().sum())


def test_tie_outcome_flat_matrix_known_counts():
    ft = np.full((2, 2), 0.25)
    r = engine.tie_outcome(ft, 0, 0)
    assert r["win"] == pytest.approx(0.25)
    assert r["level"] == pytest.approx(0.50)
    assert r["lose"] == pytest.approx(0.25)


def test_tie_outcome_away_goal_rule_is_not_applied():
    """Second leg 1-0 after a 0-1 first leg is level, not a home win."""
    ft = np.full((2, 2), 0.25)
    r = engine.tie_outcome(ft, 0, 1)   # away won the first leg 1-0
    assert r["win"] == pytest.approx(0.0)
    assert r["level"] == pytest.approx(0.25)
    assert r["lose"] == pytest.approx(0.75)


def test_tie_outcome_first_leg_lead_dominates():
    """A bigger first-leg lead only ever helps; a 2-0 on a flat grid is certain."""
    ft = np.full((2, 2), 0.25)
    r2 = engine.tie_outcome(ft, 2, 0)
    assert r2["win"] == pytest.approx(1.0)
    assert r2["lose"] == pytest.approx(0.0)
    assert r2["level"] == pytest.approx(0.0)

    m = _square(13)
    base = engine.tie_outcome(m, 0, 0)
    up = engine.tie_outcome(m, 4, 0)
    down = engine.tie_outcome(m, 0, 4)
    assert up["win"] > base["win"] > down["win"]
    assert up["lose"] < base["lose"] < down["lose"]


# ------------------------------------------------------------- fast: routing

def test_domestic_second_tier_is_not_a_cross_league_tie():
    """§5.3 pitfall: an unloaded domestic division must not hit the bridge."""
    from predictor.cli import _is_domestic_code

    # Real division codes from the fixtures feed, including the second tiers.
    assert _is_domestic_code("E0") and _is_domestic_code("E1")
    assert _is_domestic_code("E2") and _is_domestic_code("E3")
    assert _is_domestic_code("EC")
    assert _is_domestic_code("I2")       # Serie B: domestic, just not loaded
    assert _is_domestic_code("SP2")      # Segunda
    assert _is_domestic_code("D2")       # 2. Bundesliga
    assert _is_domestic_code("F2")       # Ligue 2
    assert _is_domestic_code("SC1") and _is_domestic_code("SC2")
    assert _is_domestic_code("SC3")
    assert not _is_domestic_code("CAFCL")
    assert not _is_domestic_code("CAFCC")
    assert not _is_domestic_code("UCL")
    assert not _is_domestic_code("EL")
    assert not _is_domestic_code("FAC")
    assert not _is_domestic_code("EFL")


# ------------------------------------------------------------- fast: honesty

def test_calibration_is_honest_on_synthetic_perfect_data():
    """Predicted vs realised frequency should track within a few points."""
    rng = np.random.default_rng(3)
    n = 4000
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p).astype(int)
    tab = backtest.calibration(pd.DataFrame({"p": p, "y": y}), "p", "y", bins=10)
    for _, row in tab.iterrows():
        if row["n"] >= 50:
            assert abs(row["realised"] - row["predicted"]) < 0.06, row.to_dict()


def test_known_at_schema_is_enforced():
    """§5.5: no known_at, no point-in-time query - a schema error, not a guess."""
    tbl = pd.DataFrame({"Date": [pd.Timestamp("2026-01-01")]})
    with pytest.raises(ValueError):
        loader.as_of(tbl, pd.Timestamp("2026-01-02"))


def test_known_at_answers_kickoff_minus_60():
    kickoff = pd.Timestamp("2026-09-13 20:45")
    known = pd.DataFrame({
        "Div": ["E0", "E0", "E0"],
        "Date": [pd.Timestamp("2026-09-06"), pd.Timestamp("2026-09-13"),
                 pd.Timestamp("2026-09-13")],
        "known_at": [pd.Timestamp("2026-09-06 22:00"),
                     pd.Timestamp("2026-09-13 19:00"),
                     pd.Timestamp("2026-09-13 20:10")]})  # T-60: only 19:00 made it
    snap = loader.as_of(known, kickoff - pd.Timedelta(minutes=60))
    assert list(snap["known_at"]) == [pd.Timestamp("2026-09-06 22:00"),
                                      pd.Timestamp("2026-09-13 19:00")]


def test_known_at_fixtures_are_knowable_from_read_time(tmp_path):
    """A fixture downloaded today was not knowable yesterday."""
    from predictor import fixtures

    today = pd.Timestamp(datetime.now())
    path = tmp_path / "fx.csv"
    pd.DataFrame({"Div": ["E0", "E0"],
                  "Date": [(today - pd.Timedelta(days=1)).strftime("%d/%m/%Y"),
                           (today + pd.Timedelta(days=1)).strftime("%d/%m/%Y")],
                  "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"]}).to_csv(
                      path, index=False)
    fx = fixtures.from_csv(str(path))
    assert (fx["known_at"] - today).abs().max() < pd.Timedelta(minutes=5)
    assert loader.as_of(fx, today - pd.Timedelta(days=1)).empty


# ----------------------------------------------------------------- slow: §8

@skip_slow
@pytest.mark.slow
def test_walk_forward_acceptance_constants():
    """§8: 1X2 log-loss model-alone vs blended vs price.

    Re-pinned on the Sep-2026 data refresh: four full seasons (2023/24-26/27)
    replaced the original three-season snapshot, so the pool grew from ~5,948
    to 8,239 matches and every number below moved. Values measured by
    scratch/rebase_acceptance.py.
    """
    df = loader.load(DATA)
    assert (df["known_at"] == df["Date"]).all()   # results: knowable from date on
    rows = prior_scan(df)
    assert 8000 <= len(rows) <= 8600              # ~8,200 matches
    s = score_blends(rows)
    at = s.set_index("w")
    assert at.loc[0.0, "ll1x2"] == pytest.approx(0.9802, abs=0.004)
    assert at.loc[0.0, "rps"] == pytest.approx(0.1979, abs=0.003)
    assert at.loc[0.0, "acc"] == pytest.approx(0.524, abs=0.01)
    assert at.loc[0.0, "o25"] == pytest.approx(0.6778, abs=0.005)
    assert at.loc[0.0, "btts"] == pytest.approx(0.6877, abs=0.005)
    assert at.loc[0.9, "ll1x2"] == pytest.approx(0.9616, abs=0.004)
    assert at.loc[0.9, "rps"] == pytest.approx(0.1924, abs=0.003)
    assert at.loc[0.9, "acc"] == pytest.approx(0.541, abs=0.01)
    assert at.loc[0.9, "o25"] == pytest.approx(0.6702, abs=0.005)
    assert at.loc[1.0, "ll1x2"] == pytest.approx(0.9610, abs=0.004)
    assert at.loc[1.0, "rps"] == pytest.approx(0.1923, abs=0.003)
    # The uncomfortable result, stated plainly: model adds nothing to the price.
    assert at.loc[1.0, "ll1x2"] <= at.loc[0.0, "ll1x2"]

    # §2: calibration tables track within a couple of points per bucket.
    def _ph(lam, mu, rho):
        r = markets.result(model.score_matrix_from_rates(lam, mu, rho))
        return r["H"]
    b = pd.DataFrame({"pH": [_ph(lam, mu, rho)
                             for lam, mu, rho
                             in zip(rows["lam"], rows["mu"], rows["rho"])]})
    b["hw"] = (rows["outcome"] == 0).astype(int)
    tab = backtest.calibration(b, "pH", "hw", bins=10)
    for _, row in tab.iterrows():
        if row["n"] >= 100:
            assert abs(row["realised"] - row["predicted"]) < 0.06, row.to_dict()


@skip_slow
@pytest.mark.slow
def test_uefa_bridge_acceptance():
    """§8: held-out UEFA ties ~1.010 -> ~0.933, accuracy ~50.8% -> ~58.7%.

    Re-pinned Sep-2026 after data refresh: 609 cross-league ties (was 439);
    2023-24/24-25 seasons landed entirely in train (cut = 2025-08-01) so the
    held-out test set stayed at 179. Measured by scratch/rebase_acceptance.py.
    """
    from predictor import euro
    from predictor.engine import Predictor

    p = Predictor(DATA)
    df = euro.load(os.path.join(REPO, "data", "euro"))
    cr = df.dropna(subset=["home_div", "away_div"])
    cr = cr[cr.home_div != cr.away_div]
    good = euro.resolve(cr, p)
    good = good[good.home_ok & good.away_ok]
    ds = euro.build_dataset(good, p, min_train=120)
    assert len(ds) == 609
    cut = pd.Timestamp("2025-08-01")
    tr, te = ds[ds.Date < cut], ds[ds.Date >= cut]
    assert len(te) == 179
    fit = euro.fit_offsets(tr)

    def score(d_, off):
        s = off["offsets"] if off else {}
        c = off["intercept"] if off else 0.0
        adv = off["home_adv"] if off else 0.25
        ll, hit = [], []
        for _, x in d_.iterrows():
            lvl = 0.5 * (x.h_base + x.a_base)
            sh, sa = s.get(x.home_div, 0.0), s.get(x.away_div, 0.0)
            lam = np.exp(c + lvl + x.h_att + x.a_def + sh - sa + adv)
            mu = np.exp(c + lvl + x.a_att + x.h_def + sa - sh)
            m = model.score_matrix_from_rates(lam, mu, -0.03)
            pr = markets.result(m)
            p = np.array([pr["H"], pr["D"], pr["A"]])
            o = 0 if x.FTHG > x.FTAG else (1 if x.FTHG == x.FTAG else 2)
            ll.append(-np.log(max(p[o], 1e-9)))
            hit.append(int(p.argmax() == o))
        return float(np.mean(ll)), float(np.mean(hit))

    base, base_acc = score(te, None)
    done, done_acc = score(te, fit)
    assert base == pytest.approx(1.0103, abs=0.005)
    assert done == pytest.approx(0.9328, abs=0.005)
    assert base_acc == pytest.approx(0.508, abs=0.02)
    assert done_acc == pytest.approx(0.587, abs=0.02)


@skip_slow
@pytest.mark.slow
def test_caf_bridge_acceptance():
    """§8: held-out CAF ties 0.9784 -> 0.9216 at edge_scale 1.1."""
    from predictor import euro
    from predictor.engine import Predictor

    p = Predictor(DATA)
    caf = euro.load_caf(os.path.join(REPO, "data", "caf"))
    ds, _ = euro.build_dataset_caf(caf, p)
    assert len(ds) == 199
    cut = pd.Timestamp("2024-07-01")
    tr, te = ds[ds.Date < cut], ds[ds.Date >= cut]
    assert len(te) == 124

    def score(d_, off, es):
        s = off["offsets"] if off else {}
        c = off["intercept"] if off else 0.0
        adv = off["home_adv"] if off else 0.30
        beta = off.get("beta", 1.0) if off else 1.0
        ll, hit = [], []
        for _, x in d_.iterrows():
            lvl = 0.5 * (x.h_base + x.a_base)
            sh, sa = s.get(x.home_div, 0.0), s.get(x.away_div, 0.0)
            a = c + lvl + beta * (x.h_att + x.a_def) + sh - sa + adv
            b = c + lvl + beta * (x.a_att + x.h_def) + sa - sh
            L, E = 0.5 * (a + b), 0.5 * (a - b)
            m = model.score_matrix_from_rates(
                np.exp(L + es * E), np.exp(L - es * E), -0.05)
            pr = markets.result(m)
            p = np.array([pr["H"], pr["D"], pr["A"]])
            o = 0 if x.FTHG > x.FTAG else (1 if x.FTHG == x.FTAG else 2)
            ll.append(-np.log(max(p[o], 1e-9)))
            hit.append(int(p.argmax() == o))
        return float(np.mean(ll)), float(np.mean(hit))

    base = score(te, None, 1.0)
    fit = euro.fit_offsets(tr, ridge=0.05, fit_beta=False)
    done = score(te, fit, 1.1)
    assert base[0] == pytest.approx(0.9784, abs=0.005)
    assert done[0] == pytest.approx(0.9216, abs=0.005)
    assert done[0] < base[0]