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

_SIDES = ("1", "X", "2")


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
