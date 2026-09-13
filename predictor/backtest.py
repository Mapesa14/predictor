"""Walk-forward evaluation: fit on the past only, score the next matches."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import leagues, market, markets, model

RESULTS = ["H", "D", "A"]


def rps(probs, outcome_idx) -> float:
    """Ranked probability score - the standard accuracy measure for 1X2."""
    c = np.cumsum(probs)
    o = np.zeros(len(probs))
    o[outcome_idx] = 1.0
    oc = np.cumsum(o)
    return float(np.sum((c - oc) ** 2) / (len(probs) - 1))


def walk_forward(df, div: str, xi: float = 0.0018, min_train: int = 200,
                 refit_days: int = 7, windows=("FT",),
                 goal_shrink: float = 1.0, edge_scale: float = 1.0,
                 weights: dict | None = None,
                 market_weight: float = 0.0) -> pd.DataFrame:
    """Predict every match in `div` using only matches played before it."""
    d = df[df["Div"] == div].sort_values("Date").reset_index(drop=True)
    if len(d) <= min_train:
        return pd.DataFrame()

    rows, fitted, last_fit = [], None, None
    for _, mt in d.iloc[min_train:].iterrows():
        date = mt["Date"]
        if fitted is None or (date - last_fit).days >= refit_days:
            try:
                fitted = {w: model.fit_blended(df, div, w, xi=xi, as_of=date,
                                               goal_shrink=goal_shrink,
                                               edge_scale=edge_scale,
                                               weights=weights)
                          for w in windows}
            except Exception:
                continue
            last_fit = date
        fm = fitted["FT"]
        if not (fm.knows(mt["HomeTeam"]) and fm.knows(mt["AwayTeam"])):
            continue
        lam, mu = fm.rates(mt["HomeTeam"], mt["AwayTeam"])
        # The model's own view is always kept: value has to be judged against
        # the price by something that has not already read it.
        pm = model.score_matrix_from_rates(lam, mu, fm.rho)
        pr = markets.result(pm)
        mk = None
        if market_weight > 0:
            try:
                mk = market.rates_from_row(mt, fm.rho, guess=(lam, mu))
            except Exception:
                mk = None
        blam, bmu = market.blend_rates((lam, mu), mk, market_weight)
        m = model.score_matrix_from_rates(blam, bmu, fm.rho)
        r = markets.result(m)
        tot = markets.totals(m, (2.5,))[2.5]
        bt = markets.btts(m)
        rows.append({
            "Div": div, "Date": date, "Home": mt["HomeTeam"], "Away": mt["AwayTeam"],
            "pH": r["H"], "pD": r["D"], "pA": r["A"],
            "pOver25": tot["over"], "pBTTS": bt["yes"],
            "mH": pr["H"], "mD": pr["D"], "mA": pr["A"],
            "expH": blam, "expA": bmu, "mExpH": lam, "mExpA": mu,
            "FTR": mt["FTR"], "goals": int(mt["FTHG"] + mt["FTAG"]),
            "FTHG": int(mt["FTHG"]), "FTAG": int(mt["FTAG"]),
            "over25": int(mt["FTHG"] + mt["FTAG"] > 2.5),
            "btts": int(mt["FTHG"] > 0 and mt["FTAG"] > 0),
            "AvgH": mt.get("AvgH"), "AvgD": mt.get("AvgD"), "AvgA": mt.get("AvgA"),
        })
    return pd.DataFrame(rows)


def prior_scan(df, divs=None, xi: float = 0.0018, min_train: int = 180,
               refit_days: int = 7, goal_shrink: float = 0.40,
               edge_scale: float = 1.10, weights: dict | None = None
               ) -> pd.DataFrame:
    """Walk-forward pass storing each match's model rates and the price's rates.

    This is the raw material for the market-prior sweep of §1.7 and for
    reproducing the measured acceptance constants: blends are evaluated at any
    number of weights afterwards for free, because the two rates are kept apart.
    """
    divs = divs if divs is not None else leagues.TOP_10
    weights = weights if weights is not None else {"goals": 1.0, "sot": 1.0}
    rows = []
    for div in divs:
        dd = df[df["Div"] == div].sort_values("Date").reset_index(drop=True)
        fitted, lastfit = None, None
        for _, mt in dd.iloc[min_train:].iterrows():
            if fitted is None or (mt.Date - lastfit).days >= refit_days:
                try:
                    fitted = model.fit_blended(
                        df, div, "FT", xi=xi, as_of=mt.Date,
                        goal_shrink=goal_shrink, edge_scale=edge_scale,
                        weights=weights)
                except Exception:
                    continue
                lastfit = mt.Date
            if not (fitted.knows(mt.HomeTeam) and fitted.knows(mt.AwayTeam)):
                continue
            lam, mu = fitted.rates(mt.HomeTeam, mt.AwayTeam)
            mk = market.rates_from_row(mt, fitted.rho, guess=(lam, mu))
            if mk is None:
                continue
            rows.append({
                "rho": fitted.rho, "lam": lam, "mu": mu,
                "mlam": mk[0], "mmu": mk[1],
                "outcome": {"H": 0, "D": 1, "A": 2}[mt["FTR"]],
                "over25": int(mt["FTHG"] + mt["FTAG"] > 2.5),
                "btts": int(mt["FTHG"] > 0 and mt["FTAG"] > 0)})
    return pd.DataFrame(rows)


def score_blends(rows: pd.DataFrame,
                 weights=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                 ) -> pd.DataFrame:
    """Evaluate each market weight on a prior-scan frame: the sweep of §1.7.

    Weights blend the model rate with the de-vigged price's implied rate in
    log space; 0 is model alone, 1 is price alone. The model's own rates are
    the `lam`/`mu` columns - the price's are `mlam`/`mmu` - so a weight of 0 is
    never contaminated by the price, which is what makes value-betting honest.
    """
    out = []
    for w in weights:
        P, O, B = [], [], []
        for _, x in rows.iterrows():
            lam = np.exp((1 - w) * np.log(x.lam) + w * np.log(x.mlam))
            mu = np.exp((1 - w) * np.log(x.mu) + w * np.log(x.mmu))
            m = model.score_matrix_from_rates(lam, mu, x.rho)
            res = markets.result(m)
            P.append([res["H"], res["D"], res["A"]])
            O.append(markets.totals(m, (2.5,))[2.5]["over"])
            B.append(markets.btts(m)["yes"])
        P = np.array(P)
        idx = rows.outcome.to_numpy()
        ll = -np.mean(np.log(np.clip(P[np.arange(len(P)), idx], 1e-9, 1)))
        rp = np.mean([rps(P[i], idx[i]) for i in range(len(P))])
        acc = np.mean(P.argmax(1) == idx)
        out.append({"w": w, "ll1x2": float(ll), "rps": float(rp),
                    "acc": float(acc),
                    "o25": float(_binary_ll(O, rows.over25)),
                    "btts": float(_binary_ll(B, rows.btts))})
    return pd.DataFrame(out)


def score(bt: pd.DataFrame) -> dict:
    """Summarise a walk-forward frame into headline accuracy metrics."""
    if bt.empty:
        return {}
    p = bt[["pH", "pD", "pA"]].to_numpy()
    idx = bt["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    hit = np.clip(p[np.arange(len(p)), idx], 1e-9, 1)
    out = {
        "n": len(bt),
        "logloss_1x2": float(-np.mean(np.log(hit))),
        "rps": float(np.mean([rps(p[i], idx[i]) for i in range(len(p))])),
        "acc": float(np.mean(p.argmax(1) == idx)),
        "logloss_ou25": _binary_ll(bt["pOver25"], bt["over25"]),
        "logloss_btts": _binary_ll(bt["pBTTS"], bt["btts"]),
        "brier_ou25": float(np.mean((bt["pOver25"] - bt["over25"]) ** 2)),
    }
    # A frame need not carry prices at all: the published record holds the
    # de-vigged probabilities instead, and African divisions have no price in
    # the first place. Missing columns mean no market comparison, not an error.
    if not {"AvgH", "AvgD", "AvgA"}.issubset(bt.columns):
        return out
    odds = bt[["AvgH", "AvgD", "AvgA"]].apply(pd.to_numeric, errors="coerce")
    ok = odds.notna().all(axis=1)
    if ok.sum() > 50:
        o = odds[ok].to_numpy()
        imp = np.array([market.devig(row, "shin") for row in o])
        j = idx[ok.to_numpy()]
        ihit = np.clip(imp[np.arange(len(imp)), j], 1e-9, 1)
        out["logloss_market"] = float(-np.mean(np.log(ihit)))
        out["rps_market"] = float(np.mean([rps(imp[i], j[i]) for i in range(len(imp))]))
        out["n_with_odds"] = int(ok.sum())
    return out


def _binary_ll(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(bt: pd.DataFrame, col: str, actual: str, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised frequency, bucketed - the honesty check."""
    b = bt.copy()
    b["bin"] = pd.cut(b[col], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = b.groupby("bin", observed=True).agg(
        n=(col, "size"), predicted=(col, "mean"), realised=(actual, "mean"))
    return g.reset_index()


def value_bets(bt: pd.DataFrame, edge: float = 0.05, kelly_cap: float = 0.05) -> pd.DataFrame:
    """Flat-stake and Kelly returns on 1X2 selections the model rates above price."""
    b = bt.dropna(subset=["AvgH", "AvgD", "AvgA"]).copy()
    if b.empty:
        return pd.DataFrame()
    # Model-only probabilities where present: a forecast that already contains
    # the price cannot be used to find value against it.
    hcol, dcol, acol = (("mH", "mD", "mA") if "mH" in b.columns
                        else ("pH", "pD", "pA"))
    picks = []
    for _, r in b.iterrows():
        for sel, pcol, ocol in ((("H", hcol, "AvgH"), ("D", dcol, "AvgD"),
                                 ("A", acol, "AvgA"))):
            price = float(r[ocol])
            p = float(r[pcol])
            ev = p * price - 1.0
            if ev >= edge:
                won = r["FTR"] == sel
                k = max(0.0, min(kelly_cap, (p * price - 1) / (price - 1)))
                picks.append({"Date": r["Date"], "Div": r["Div"],
                              "match": r["Home"] + " v " + r["Away"], "sel": sel,
                              "p": p, "price": price, "ev": ev, "won": bool(won),
                              "flat_pnl": (price - 1) if won else -1.0,
                              "kelly_stake": k,
                              "kelly_pnl": k * ((price - 1) if won else -1.0)})
    return pd.DataFrame(picks)
