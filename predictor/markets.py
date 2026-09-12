"""Turn a scoreline probability matrix into every common betting market."""
from __future__ import annotations

import numpy as np

HT_FT_KEYS = [("H", "H"), ("H", "D"), ("H", "A"),
              ("D", "H"), ("D", "D"), ("D", "A"),
              ("A", "H"), ("A", "D"), ("A", "A")]


# --------------------------------------------------------------- primitives
def _grid(m):
    n = m.shape[0]
    h = np.arange(n)[:, None] * np.ones((1, n))
    a = np.ones((n, 1)) * np.arange(n)[None, :]
    return h, a


def result(m) -> dict:
    h, a = _grid(m)
    return {"H": float(m[h > a].sum()), "D": float(np.trace(m)), "A": float(m[h < a].sum())}


def double_chance(m) -> dict:
    r = result(m)
    return {"1X": r["H"] + r["D"], "12": r["H"] + r["A"], "X2": r["D"] + r["A"]}


def draw_no_bet(m) -> dict:
    r = result(m)
    live = r["H"] + r["A"]
    return {"H": r["H"] / live, "A": r["A"] / live} if live > 0 else {"H": .5, "A": .5}


def totals(m, lines=(0.5, 1.5, 2.5, 3.5, 4.5, 5.5)) -> dict:
    h, a = _grid(m)
    tot = h + a
    out = {}
    for ln in lines:
        over = float(m[tot > ln].sum())
        out[ln] = {"over": over, "under": 1.0 - over}
    return out


def team_totals(m, lines=(0.5, 1.5, 2.5, 3.5)) -> dict:
    h, a = _grid(m)
    out = {"home": {}, "away": {}}
    for ln in lines:
        oh = float(m[h > ln].sum())
        oa = float(m[a > ln].sum())
        out["home"][ln] = {"over": oh, "under": 1.0 - oh}
        out["away"][ln] = {"over": oa, "under": 1.0 - oa}
    return out


def btts(m) -> dict:
    h, a = _grid(m)
    yes = float(m[(h > 0) & (a > 0)].sum())
    return {"yes": yes, "no": 1.0 - yes}


def clean_sheet(m) -> dict:
    return {"home": float(m[:, 0].sum()), "away": float(m[0, :].sum())}


def win_to_nil(m) -> dict:
    return {"home": float(m[1:, 0].sum()), "away": float(m[0, 1:].sum())}


def odd_even(m) -> dict:
    h, a = _grid(m)
    odd = float(m[((h + a) % 2 == 1)].sum())
    return {"odd": odd, "even": 1.0 - odd}


def correct_scores(m, top: int = 8) -> list:
    idx = np.dstack(np.unravel_index(np.argsort(m, axis=None)[::-1], m.shape))[0]
    return [(int(i), int(j), float(m[i, j])) for i, j in idx[:top]]


def winning_margin(m) -> dict:
    h, a = _grid(m)
    d = h - a
    out = {}
    for k in (1, 2, 3):
        out["home_by_%d" % k] = float(m[d == k].sum())
        out["away_by_%d" % k] = float(m[d == -k].sum())
    out["home_by_4+"] = float(m[d >= 4].sum())
    out["away_by_4+"] = float(m[d <= -4].sum())
    out["draw"] = float(np.trace(m))
    return out


def european_handicap(m, lines=(-2, -1, 1, 2)) -> dict:
    """Whole-goal handicap applied to the home team, with the draw still live."""
    h, a = _grid(m)
    out = {}
    for ln in lines:
        d = (h + ln) - a
        out[ln] = {"home": float(m[d > 0].sum()),
                   "draw": float(m[d == 0].sum()),
                   "away": float(m[d < 0].sum())}
    return out


def asian_handicap(m, lines=(-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
                             -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)) -> dict:
    """Win probability for each side; a pushed quarter returns that half stake."""
    h, a = _grid(m)
    diff = h - a
    out = {}
    for ln in lines:
        parts = [ln] if (ln * 4) % 2 == 0 else [ln - 0.25, ln + 0.25]
        hw = hl = 0.0
        for p in parts:
            adj = diff + p
            win = float(m[adj > 0].sum())
            lose = float(m[adj < 0].sum())
            push = float(m[adj == 0].sum())
            live = win + lose
            # A push returns the stake, so spread it over the settled outcomes.
            hw += (win + push * (win / live if live else .5)) / len(parts)
            hl += (lose + push * (lose / live if live else .5)) / len(parts)
        s = hw + hl
        out[ln] = {"home": hw / s, "away": hl / s}
    return out


def _total_dist(m) -> np.ndarray:
    n = m.shape[0]
    out = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            out[i + j] += m[i, j]
    return out


def half_with_most_goals(first: np.ndarray, second: np.ndarray) -> dict:
    f = _total_dist(first)
    s = _total_dist(second)
    j = np.outer(f, s)
    i = np.arange(len(f))
    g = i[:, None] - i[None, :]
    return {"first": float(j[g > 0].sum()),
            "second": float(j[g < 0].sum()),
            "equal": float(j[g == 0].sum())}


def ht_ft(first: np.ndarray, second: np.ndarray, max_goals: int = 7) -> dict:
    """Joint half-time / full-time distribution built from the two half models."""
    n = min(max_goals + 1, first.shape[0])
    f = first[:n, :n]
    s = second[:n, :n]
    out = {k: 0.0 for k in HT_FT_KEYS}
    for h1 in range(n):
        for a1 in range(n):
            p1 = f[h1, a1]
            if p1 < 1e-10:
                continue
            ht = "H" if h1 > a1 else ("A" if h1 < a1 else "D")
            for h2 in range(n):
                for a2 in range(n):
                    p = p1 * s[h2, a2]
                    if p < 1e-12:
                        continue
                    fh, fa = h1 + h2, a1 + a2
                    ft = "H" if fh > fa else ("A" if fh < fa else "D")
                    out[(ht, ft)] += p
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def combined(m) -> dict:
    """Popular combinations: result + over/under 2.5, and result + BTTS."""
    h, a = _grid(m)
    tot = h + a
    res = {"H": h > a, "D": h == a, "A": h < a}
    both = (h > 0) & (a > 0)
    out = {}
    for k, mask in res.items():
        out[k + "&O2.5"] = float(m[mask & (tot > 2.5)].sum())
        out[k + "&U2.5"] = float(m[mask & (tot < 2.5)].sum())
        out[k + "&BTTS"] = float(m[mask & both].sum())
    return out


def score_grid(ft: np.ndarray, max_goals: int = 5) -> list[list[float]]:
    """The 0..max_goals x 0..max_goals slice of one matrix, renormalised.

    Same shape as the card's signature graphic: rows are home goals, columns
    away goals, cell value the (rounded) probability of that exact scoreline.
    """
    g = ft[: max_goals + 1, : max_goals + 1]
    g = np.asarray(g, dtype=float)
    g /= max(float(g.sum()), 1e-12)
    return [[round(float(v), 6) for v in row] for row in g]


def summary(ft: np.ndarray, first=None, second=None) -> dict:
    """Every market derived from one fixture, in a single dictionary."""
    h, a = _grid(ft)
    out = {
        "result": result(ft),
        "double_chance": double_chance(ft),
        "draw_no_bet": draw_no_bet(ft),
        "totals": totals(ft),
        "team_totals": team_totals(ft),
        "btts": btts(ft),
        "clean_sheet": clean_sheet(ft),
        "win_to_nil": win_to_nil(ft),
        "odd_even": odd_even(ft),
        "correct_scores": correct_scores(ft, 10),
        "winning_margin": winning_margin(ft),
        "asian_handicap": asian_handicap(ft),
        "european_handicap": european_handicap(ft),
        "combined": combined(ft),
        "exp_home": float((h * ft).sum()),
        "exp_away": float((a * ft).sum()),
    }
    if first is not None:
        fh, fa = _grid(first)
        out["ht_result"] = result(first)
        out["ht_totals"] = totals(first, (0.5, 1.5, 2.5))
        out["ht_btts"] = btts(first)
        out["ht_exp_home"] = float((fh * first).sum())
        out["ht_exp_away"] = float((fa * first).sum())
    if first is not None and second is not None:
        out["ht_ft"] = ht_ft(first, second)
        out["half_most_goals"] = half_with_most_goals(first, second)
    return out
