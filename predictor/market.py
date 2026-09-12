"""Bookmaker prices, turned into goal rates the model can be blended with.

The closing line is a better forecast than this model, measurably so. Rather
than compete with it, the price is read back into the same two numbers the
model produces - a home and an away goal rate - so it can act as a prior.

The pure-model probabilities are always kept alongside the blended ones. Once
the price is inside the forecast, comparing that forecast to the price is
circular, and any "value" it reports is an illusion.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import markets, model

# Column trios in the football-data files, best consensus first.
ODDS_1X2 = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"),
            ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
ODDS_OU25 = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"),
             ("Max>2.5", "Max<2.5")]


# ----------------------------------------------------------------- de-vigging
def devig_proportional(odds) -> np.ndarray:
    """Strip the margin by scaling every implied probability equally."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def devig_shin(odds, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Shin's method: assumes the margin comes from insider money.

    Takes more margin off short prices than long ones, which matches how books
    actually price the favourite-longshot bias.
    """
    inv = 1.0 / np.asarray(odds, dtype=float)
    booksum = inv.sum()
    if booksum <= 1.0 or len(inv) < 2:
        return inv / booksum
    z = 0.0
    for _ in range(max_iter):
        root = np.sqrt(z * z + 4.0 * (1.0 - z) * inv * inv / booksum)
        p = (root - z) / (2.0 * (1.0 - z))
        s = p.sum()
        if abs(s - 1.0) < tol:
            break
        # one damped Newton-ish step on z
        z = np.clip(z + (s - 1.0) * 0.5, 0.0, 0.4)
    p = np.clip(p, 1e-9, None)
    return p / p.sum()


def devig(odds, method: str = "shin") -> np.ndarray:
    return devig_shin(odds) if method == "shin" else devig_proportional(odds)


def overround(odds) -> float:
    """The bookmaker's margin, as a fraction over a fair book."""
    return float(np.sum(1.0 / np.asarray(odds, dtype=float)) - 1.0)


# ------------------------------------------------------- price -> goal rates
def implied_rates(p_home: float, p_draw: float, p_away: float, rho: float = 0.0,
                  p_over25: float | None = None,
                  guess: tuple[float, float] = (1.4, 1.2),
                  max_goals: int = 10) -> tuple[float, float]:
    """The goal rates whose scoreline matrix best reproduces these prices.

    Solving in the model's own currency keeps the one-matrix design intact: the
    blended forecast is still a single distribution, so no two markets on the
    card can contradict each other.
    """
    target = np.log(np.clip([p_home, p_draw, p_away], 1e-6, 1.0))

    def residual(x):
        lam, mu = np.exp(x)
        m = model.score_matrix_from_rates(lam, mu, rho, max_goals)
        r = markets.result(m)
        out = np.log(np.clip([r["H"], r["D"], r["A"]], 1e-9, 1.0)) - target
        if p_over25 is not None:
            got = markets.totals(m, (2.5,))[2.5]["over"]
            out = np.append(out, np.log(np.clip(got, 1e-9, 1.0))
                            - np.log(max(p_over25, 1e-6)))
        return out

    res = least_squares(residual, np.log(np.asarray(guess, dtype=float)),
                        bounds=(np.log([0.03, 0.03]), np.log([9.0, 9.0])),
                        xtol=1e-8, ftol=1e-8, max_nfev=200)
    lam, mu = np.exp(res.x)
    return float(lam), float(mu)


def _pick(row, groups):
    """First complete odds group present in the row."""
    for cols in groups:
        try:
            vals = [float(row[c]) for c in cols]
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(v) and v > 1.0 for v in vals):
            return vals
    return None


def rates_from_row(row, rho: float = 0.0, guess=(1.4, 1.2),
                   method: str = "shin", use_totals: bool = True):
    """Read a fixture row's prices into a pair of goal rates, or None."""
    o = _pick(row, ODDS_1X2)
    if o is None:
        return None
    p = devig(o, method)
    p_over = None
    if use_totals:
        ou = _pick(row, ODDS_OU25)
        if ou is not None:
            p_over = float(devig(ou, method)[0])
    return implied_rates(p[0], p[1], p[2], rho, p_over, guess)


def blend_rates(model_rates, market_rates, weight: float):
    """Combine two rate pairs in log space. weight is the market's share."""
    if market_rates is None or weight <= 0:
        return model_rates
    w = min(max(weight, 0.0), 1.0)
    lam = np.exp((1 - w) * np.log(model_rates[0]) + w * np.log(market_rates[0]))
    mu = np.exp((1 - w) * np.log(model_rates[1]) + w * np.log(market_rates[1]))
    return float(lam), float(mu)
