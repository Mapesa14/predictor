"""Clear picks, chosen by a fixed rule and measured before being offered.

Two lists and one warning:

  Bankers    the short list: up to 8 fixtures whose favourite is 75%+ and
             whose closing price backs the same side.
  Long list  up to 20 at 65%+, price agreeing.
  Avoid      fixtures where the model and the price back different sides.

What these are and are not, measured walk-forward over 8,039 matches in the
ten top European leagues (scratch/tips.py, 2026-09-13), every prediction made
only from matches played before it:

  * Confidence filters for *accuracy*, and stably: 75%+ favourites won 84.0%,
    84.3% and 84.6% across three seasons.
  * It is not an edge. Flat-stake return sat between -2.5% and +0.3% at every
    threshold. Short-priced favourites win often because they are priced to.
  * A shortlist is not a safe accumulator. The week's 4 most confident picks
    all won in 47.6% of weeks, 8 in 16.5%. The product of the model's own
    probabilities predicted that closely (45.2% for 4), so the list shows that
    number instead of letting anyone read "sure".
  * Disagreeing with the price is where the model is worst: 121 picks, 29.8%
    won, -18.1% flat return. Those go on the avoid list, never on a tip list.

Rules that exist so the lists cannot flatter themselves:

  * never padded - if three fixtures clear the bar, the short list has three
    and says so, rather than lowering the bar to reach four;
  * nothing already under way;
  * nothing unpriced in either list. Tanzanian and CAF fixtures have no closing
    price to agree with and no benchmark yet, so they get their own section
    until the published record says otherwise;
  * at most three bankers from one league - otherwise the top of the table is
    mostly Portugal. A spread rule, not an accuracy rule, and reported when it
    bites;
  * a draw is never tipped as a 1X2 pick.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import leagues

BANKER_MIN = 0.75
LONG_MIN = 0.65
BANKER_MAX = 8
LONG_MAX = 20
BANKER_TARGET = 4          # below this the short list says so rather than padding
LONG_TARGET = 10
LEAGUE_CAP = 3             # bankers only
MIN_JUDGEABLE = 30         # settled picks before a record hit rate means much

# The measurement these rules rest on. Reported with every list so a number on
# screen always carries where it came from. Refresh by re-running
# scratch/tips.py; test_tips pins the thresholds here to the ones in use.
EVIDENCE = {
    "source": ("walk-forward over 8,039 matches in the 10 top European leagues, "
               "2023/24 to 2026/27, measured 2026-09-13 (scratch/tips.py)"),
    "bankers": {"threshold": BANKER_MIN, "hit": 0.842, "per_week_median": 6,
                "flat_return": 0.003, "hit_by_season": [0.840, 0.843, 0.846]},
    "long_list": {"threshold": LONG_MIN, "hit": 0.763, "per_week_median": 16.5,
                  "flat_return": -0.013, "top10_hit": 0.791, "top20_hit": 0.732,
                  "worst_week_top10": 0.40},
    "accumulator_all_won": {"4": 0.476, "6": 0.295, "8": 0.165},
    "double_chance_all_won": {"4": 0.838, "8": 0.612},
    "disagree_with_price": {"n": 121, "hit": 0.298, "flat_return": -0.181},
}

# ---------------------------------------------------------- pick categories
# Measured the same way the 1X2 lists were, walk-forward over 8,230 matches in
# the ten top European leagues across four seasons (scratch/eval_markets.py,
# 2026-10-09): nothing has seen its own result, and every number below is what
# the rule would have delivered, not what the model hoped for.
#
# A category is offered only where the model is honest at its bar: "model said"
# is the average probability it gave those picks, "hit" what came in. Three
# candidates were measured and left out, which is the point of measuring:
#
#   under 2.5 goals  67.6% at 65%+, but 0.3 picks a matchday and -2.8% flat;
#   over 3.5 goals   model said 66.0%, 64.1% came in - optimistic and thin;
#   no GG            model said 62.5%, 57.5% came in, and one season managed
#                    46.7%. A "clear pick" that loses more often than not in a
#                    bad year is not a clear pick.
# Each leg is a selection measured in its own right: the home-win bar and the
# away-win bar are not the same number, and one averaged claim over both would
# misstate each. A category's headline is the n-weighted combination of its
# legs, and the legs stay on screen.
def _leg(sel, code, label, min_p, hit, model_said, worst, n, per_day, flat=None):
    return {"sel": sel, "code": code, "label": label, "min": min_p,
            "hit": hit, "model_said": model_said, "worst_season": worst,
            "n": n, "per_matchday": per_day, "flat_return": flat}


CATEGORIES = [
    {"key": "over05", "label": "A goal in the match", "short": "Over 0.5",
     "max": 12,
     "why": "The safest thing the model says. It is priced at about 1.05, so "
            "it is an accumulator leg rather than a bet on its own.",
     "legs": [_leg("over05", "over05", "Over 0.5 goals",
                   0.95, 0.972, 0.963, 0.970, 1876, 3.5)]},
    {"key": "over15", "label": "Over 1.5 goals", "short": "Over 1.5",
     "max": 12,
     "why": "Two goals in the match, and the best-calibrated line we have: "
            "the model said 84.8% and 84.4% came in.",
     "legs": [_leg("over15", "over15", "Over 1.5 goals",
                   0.80, 0.844, 0.848, 0.836, 3186, 5.9)]},
    {"key": "double_chance", "label": "Double chance", "short": "1X / X2",
     "max": 10,
     "why": "The favourite not to lose. Nearly nine in ten come in, and the "
            "model is if anything a shade cautious here.",
     "legs": [_leg("dc_1x", "1X", "%(home)s or draw",
                   0.80, 0.890, 0.875, 0.883, 2261, 4.2),
              _leg("dc_x2", "X2", "%(away)s or draw",
                   0.80, 0.891, 0.861, 0.881, 908, 1.7)]},
    {"key": "win", "label": "Win", "short": "1 / 2", "max": 8,
     "why": "Confidence alone. The short list above also requires the closing "
            "price to back the same side, which is the version with a "
            "published record behind it.",
     "legs": [_leg("home", "1", "%(home)s",
                   0.75, 0.848, 0.814, 0.842, 560, 1.0, 0.004),
              _leg("away", "2", "%(away)s",
                   0.70, 0.805, 0.758, 0.760, 246, 0.5, 0.025)]},
    {"key": "over25", "label": "Over 2.5 goals", "short": "Over 2.5",
     "max": 10,
     "why": "Three goals. Well calibrated, but the price knows it: a flat "
            "stake on every one of these lost 1.0% across four seasons.",
     "legs": [_leg("over25", "over25", "Over 2.5 goals",
                   0.65, 0.715, 0.709, 0.688, 1179, 2.2, -0.010)]},
    {"key": "btts", "label": "Both teams to score (GG)", "short": "GG",
     "max": 10,
     "why": "The weakest category offered, and the only one where the model "
            "flatters itself: it said 63.6% and 61.8% came in. Read these as "
            "leans, never as bankers.",
     "legs": [_leg("btts_yes", "btts_yes", "Both teams to score",
                   0.60, 0.618, 0.636, 0.613, 1444, 2.7)]},
]

# Three more were measured and left out, which is the point of measuring:
#   under 2.5 goals  67.6% at the 65% bar, but 0.3 picks a matchday and -2.8%
#                    flat - too rare to be a list, too dear to be a bet;
#   over 3.5 goals   the model said 66.0% and 64.1% came in, on 0.2 a matchday;
#   no GG            the model said 62.5%, 57.5% came in, and one season
#                    managed 46.7%. A "clear pick" that loses more often than
#                    it wins in a bad year is not a clear pick.
NOT_OFFERED = [
    {"label": "Under 2.5 goals", "hit": 0.676, "model_said": 0.675,
     "per_matchday": 0.3, "flat_return": -0.028,
     "why": "too rare to be a list, and a flat stake lost 2.8%"},
    {"label": "Over 3.5 goals", "hit": 0.641, "model_said": 0.660,
     "per_matchday": 0.2, "flat_return": None,
     "why": "the model is optimistic there, and it fires twice a week"},
    {"label": "No both teams to score", "hit": 0.575, "model_said": 0.625,
     "per_matchday": 0.6, "flat_return": None,
     "why": "the model said 62.5%, 57.5% came in, and one season managed 46.7%"},
]


def _combine(legs: list) -> dict:
    """A category's headline: its legs weighted by how often each fires."""
    n = sum(l["n"] for l in legs) or 1
    priced = [l for l in legs if l["flat_return"] is not None]
    return {
        "min": min(l["min"] for l in legs),
        "hit": round(sum(l["hit"] * l["n"] for l in legs) / n, 4),
        "model_said": round(sum(l["model_said"] * l["n"] for l in legs) / n, 4),
        "worst_season": min(l["worst_season"] for l in legs),
        "n": n, "per_matchday": round(sum(l["per_matchday"] for l in legs), 2),
        "flat_return": (round(sum(l["flat_return"] * l["n"] for l in priced)
                              / sum(l["n"] for l in priced), 4)
                        if priced else None),
        "legs": [{"label": l["label"].replace("%(home)s", "Home")
                                     .replace("%(away)s", "Away"),
                  "min": l["min"], "hit": l["hit"],
                  "model_said": l["model_said"]} for l in legs],
    }


for _c in CATEGORIES:
    _c["evidence"] = _combine(_c["legs"])

CATEGORY_MIN = {l["sel"]: l["min"] for c in CATEGORIES for l in c["legs"]}

CATEGORY_SOURCE = ("walk-forward over 8,230 matches in the 10 top European "
                   "leagues, 2023/24 to 2026/27, measured 2026-10-09 "
                   "(scratch/eval_markets.py)")


def _cat_pick(m: dict, leg: dict, now) -> dict | None:
    """One fixture as one category pick, or None if it does not clear the bar."""
    mk = m.get("markets") or {}
    p = mk.get(leg["sel"])
    if p is None:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p < leg["min"]:
        return None
    started = m.get("started")
    if started is None:
        ko = _parse(m.get("date"))
        started = bool(ko and ko <= (now or datetime.now(timezone.utc)))
    if started:
        return None
    od = m.get("odds") if isinstance(m.get("odds"), dict) else {}
    mkt = _probs(m.get("market"))
    code = leg["code"]
    # Only 1X2 carries a closing price in the feed. The goals lines and GG are
    # unpriced here, and say so rather than implying a check that never ran.
    return {"div": m.get("div"),
            "league": m.get("league") or leagues.name(m.get("div")),
            "kickoff": m.get("date"), "kickoff_label": m.get("kickoff_label"),
            "home": m.get("home"), "away": m.get("away"),
            "pick": code,
            "side": leg["label"] % {"home": m.get("home"), "away": m.get("away")},
            "p": p,
            "market_p": mkt[code] if (mkt and code in mkt) else None,
            "odds": float(od[code]) if od.get(code) is not None else None,
            "started": False}


def categories(rows, now=None) -> list[dict]:
    """Every category, its measurement, and the picks clearing its own bar."""
    now = now or datetime.now(timezone.utc)
    out = []
    for cat in CATEGORIES:
        picks = []
        for m in rows or []:
            if m.get("comp"):          # cups: rotated line-ups, no settlement
                continue
            for leg in cat["legs"]:
                c = _cat_pick(m, leg, now)
                if c:
                    picks.append(c)
        picks.sort(key=lambda c: (-c["p"], c["kickoff"] or ""))
        picks = picks[:cat["max"]]
        out.append({"key": cat["key"], "label": cat["label"],
                    "short": cat["short"], "why": cat["why"],
                    "evidence": cat["evidence"], "picks": picks,
                    "accumulator": _acca_p([c["p"] for c in picks[:4]])})
    return out


def _acca_p(ps: list) -> dict | None:
    return {"legs": len(ps), "all_win": float(np.prod(ps))} if ps else None


_SIDES = ("1", "X", "2")


# ---------------------------------------------------------- pick categories
def selections(s: dict) -> dict:
    """Every selection a tip category can offer, from one market summary.

    One place, so the slate, the CLI and the frozen lists cannot drift: a tip
    that disagrees with its own match card is the fastest way to lose a reader.
    """
    t = s.get("totals") or {}
    r = s.get("result") or {}
    b = s.get("btts") or {}
    dc = s.get("double_chance") or {}

    def line(x, side):
        v = t.get(x) or {}
        return v.get(side)

    return {"over05": line(0.5, "over"), "over15": line(1.5, "over"),
            "over25": line(2.5, "over"), "under25": line(2.5, "under"),
            "over35": line(3.5, "over"),
            "btts_yes": b.get("yes"), "btts_no": b.get("no"),
            "dc_1x": dc.get("1X"), "dc_x2": dc.get("X2"), "dc_12": dc.get("12"),
            "home": r.get("H"), "away": r.get("A")}


def _parse(ts):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _probs(d):
    if not isinstance(d, dict):
        return None
    try:
        return {k: float(d[k]) for k in _SIDES}
    except (KeyError, TypeError, ValueError):
        return None


def candidate(m: dict, now: datetime | None = None) -> dict | None:
    """One slate row as a possible tip, with everything the rules need."""
    probs = _probs(m.get("p"))
    if probs is None:
        return None
    pick = max(_SIDES, key=lambda k: probs[k])
    mk = _probs(m.get("market"))
    od = m.get("odds") if isinstance(m.get("odds"), dict) else {}
    started = m.get("started")
    if started is None:
        ko = _parse(m.get("date"))
        started = bool(ko and ko <= (now or datetime.now(timezone.utc)))
    return {
        "div": m.get("div"),
        "league": m.get("league") or leagues.name(m.get("div")),
        "kickoff": m.get("date"),
        "kickoff_label": m.get("kickoff_label"),
        "home": m.get("home"), "away": m.get("away"),
        "pick": pick,
        "side": {"1": m.get("home"), "X": "Draw", "2": m.get("away")}[pick],
        "p": probs[pick],
        "p_double_chance": probs[pick] + probs["X"] if pick != "X" else None,
        "market_p": mk[pick] if mk else None,
        # the bookmakers' average decimal price on the pick, margin included,
        # so a frozen list's flat-stake return is the one a bettor would get
        "odds": float(od[pick]) if od.get(pick) is not None else None,
        "priced": mk is not None,
        "agrees": (max(_SIDES, key=lambda k: mk[k]) == pick) if mk else None,
        "score": m.get("score"),
        "started": bool(started),
    }


def _acca(legs: list) -> dict | None:
    if not legs:
        return None
    dc = [c["p_double_chance"] for c in legs]
    return {"legs": len(legs),
            "all_win": float(np.prod([c["p"] for c in legs])),
            "all_win_double_chance": float(np.prod(dc)) if all(dc) else None}


def build(rows, now: datetime | None = None, banker_min: float = BANKER_MIN,
          long_min: float = LONG_MIN, banker_max: int = BANKER_MAX,
          long_max: int = LONG_MAX, league_cap: int | None = LEAGUE_CAP) -> dict:
    """The day's lists from the slate's own rows."""
    now = now or datetime.now(timezone.utc)
    # Cup ties are left out: rotated line-ups make a favourite's probability
    # least reliable exactly where a short list would lean on it hardest.
    cands = [c for c in (candidate(m, now) for m in rows or []
                         if not m.get("comp")) if c]
    upcoming = [c for c in cands if not c["started"]]
    upcoming.sort(key=lambda c: (-c["p"], c["kickoff"] or ""))

    tippable = [c for c in upcoming
                if c["priced"] and c["agrees"] and c["pick"] != "X"]

    bankers, per_league, capped = [], {}, 0
    for c in tippable:
        if c["p"] < banker_min or len(bankers) >= banker_max:
            break
        if league_cap and per_league.get(c["div"], 0) >= league_cap:
            capped += 1
            continue
        bankers.append(c)
        per_league[c["div"]] = per_league.get(c["div"], 0) + 1

    long_list = [c for c in tippable if c["p"] >= long_min][:long_max]
    unpriced = [c for c in upcoming
                if not c["priced"] and c["pick"] != "X" and c["p"] >= long_min]
    avoid = [c for c in upcoming if c["priced"] and c["agrees"] is False]

    notes = []
    if len(bankers) < BANKER_TARGET:
        notes.append("Only %d fixture%s in this window clear%s the %d%% bar for "
                     "the short list. It is not padded to reach %d."
                     % (len(bankers), "" if len(bankers) == 1 else "s",
                        "s" if len(bankers) == 1 else "",
                        round(banker_min * 100), BANKER_TARGET))
    if len(long_list) < LONG_TARGET:
        notes.append("The long list has %d, below the usual %d: nothing else "
                     "clears %d%% with the price agreeing."
                     % (len(long_list), LONG_TARGET, round(long_min * 100)))
    if capped:
        notes.append("%d more fixture%s cleared the short-list bar but %s left "
                     "out to keep at most %d from one league."
                     % (capped, "" if capped == 1 else "s",
                        "was" if capped == 1 else "were", league_cap))

    return {
        "categories": categories(rows, now),
        "category_source": CATEGORY_SOURCE,
        "not_offered": NOT_OFFERED,
        "bankers": bankers,
        "accumulator": _acca(bankers),
        "long_list": long_list,
        "unpriced": unpriced,
        "avoid": avoid,
        "notes": notes,
        "rules": {"banker_min": banker_min, "long_min": long_min,
                  "banker_max": banker_max, "long_max": long_max,
                  "league_cap": league_cap},
        "considered": len(upcoming),
        "excluded_started": len(cands) - len(upcoming),
        "evidence": EVIDENCE,
    }


def record_performance(settled: pd.DataFrame) -> dict:
    """How the rules have done on the published record, settled rows only.

    Applied to every prediction ever published, not to the lists as they were
    shown, so it cannot be tuned after the fact. It is the threshold rule only
    - no league cap or top-N cut - which is why it is labelled as the rule's
    record rather than the lists'.
    """
    empty = {"settled": 0}
    for name in ("bankers", "long_list", "unpriced", "avoid"):
        empty[name] = {"n": 0, "hit": None, "enough": False}
    if settled is None or not len(settled) or "FTR" not in settled.columns:
        return empty
    d = settled[settled["FTR"].notna()].copy()
    if d.empty:
        return empty

    P = d[["pH", "pD", "pA"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    idx = P.argmax(axis=1)
    pick = np.array(["H", "D", "A"])[idx]
    conf = P.max(axis=1)
    M = d[["mk1", "mkX", "mk2"]].apply(pd.to_numeric, errors="coerce")
    priced = M.notna().all(axis=1).to_numpy()
    agree = np.zeros(len(d), dtype=bool)
    if priced.any():
        agree[priced] = M.to_numpy(float)[priced].argmax(axis=1) == idx[priced]
    won = pick == d["FTR"].to_numpy()
    notdraw = pick != "D"

    groups = {
        "bankers": priced & agree & notdraw & (conf >= BANKER_MIN),
        "long_list": priced & agree & notdraw & (conf >= LONG_MIN),
        "unpriced": ~priced & notdraw & (conf >= LONG_MIN),
        "avoid": priced & ~agree,
    }
    out = {"settled": int(len(d))}
    for name, mask in groups.items():
        n = int(mask.sum())
        out[name] = {"n": n, "hit": float(won[mask].mean()) if n else None,
                     "enough": n >= MIN_JUDGEABLE}
    return out
