"""The day's tip lists, frozen before kick-off, and how they did.

`tips.build` makes the lists afresh on every request, from the fixtures and
prices of that moment, so yesterday's lists existed nowhere: "the bankers went
5 of 6" could not be said, let alone checked. This module keeps them, under the
same two rules as the public record (record.py):

  * `freeze` writes a day's lists once, at FREEZE_HOUR East Africa Time -
    before the day's first regular kick-off. A day already frozen is never
    frozen again, and a fixture that turns up later that day is not in its
    lists. What was shown is what gets judged.
  * results are never stored. `settle` joins the frozen picks to results on
    every read: first the official sources the engine fits on, then
    API-Football's final scores for what those do not carry. When the two
    disagree the official score is used and the pick is flagged.

Rows are hash-chained exactly like the record, so an edited pick shows up in
`verify`. Every day also gets a marker row, so a day on which nothing cleared
the bars is recorded as frozen-and-empty rather than looking like a gap.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import db, fixtures, record, tips

EAT = ZoneInfo("Africa/Dar_es_Salaam")
FREEZE_HOUR = int(os.environ.get("TIPS_FREEZE_HOUR", "10") or 10)
BASE_LISTS = ("bankers", "long_list", "unpriced", "avoid")
# The 1X2 lists, then one list per pick category (goals lines, double chance,
# GG). A category is frozen and judged exactly like the others.
LISTS = BASE_LISTS + tuple(c["key"] for c in tips.CATEGORIES)
DAY_MARK = "_day"

COLUMNS = ["frozen_at", "day", "list", "rank", "div", "league", "date",
           "kickoff", "home", "away", "pick", "side", "p", "p_dc", "market_p",
           "odds", "prev", "hash"]
_NUM = ["p", "p_dc", "market_p", "odds"]

FINISHED = {"FT", "AET", "PEN"}
VOID = {"PST", "CANC", "ABD", "AWD", "WO"}

# Every pick code the lists can carry, settled from the 90-minute score. The
# code lives in the log's existing `pick` column, so adding categories changed
# no column and broke no hash already written.
_SETTLE = {
    "1": lambda h, a: h > a,
    "X": lambda h, a: h == a,
    "2": lambda h, a: a > h,
    "1X": lambda h, a: h >= a,
    "X2": lambda h, a: a >= h,
    "12": lambda h, a: h != a,
    "over05": lambda h, a: h + a > 0,
    "over15": lambda h, a: h + a > 1,
    "over25": lambda h, a: h + a > 2,
    "over35": lambda h, a: h + a > 3,
    "under25": lambda h, a: h + a < 3,
    "btts_yes": lambda h, a: h > 0 and a > 0,
    "btts_no": lambda h, a: h == 0 or a == 0,
}
_1X2 = ("1", "X", "2")

# What each list was offered on: the measured hit rate in tips.EVIDENCE. The
# unpriced list has no benchmark, which is the reason it is a separate list.
PROMISED = {
    "bankers": tips.EVIDENCE["bankers"]["hit"],
    "long_list": tips.EVIDENCE["long_list"]["hit"],
    "unpriced": None,
    "avoid": tips.EVIDENCE["disagree_with_price"]["hit"],
}
PROMISED.update({c["key"]: c["evidence"]["hit"] for c in tips.CATEGORIES})

# What each list is called on screen, so a frozen row reads the same months on.
LABELS = {"bankers": "Bankers", "long_list": "Long list",
          "unpriced": "Unpriced", "avoid": "Avoid"}
LABELS.update({c["key"]: c["label"] for c in tips.CATEGORIES})


def path(root: str) -> str:
    return os.path.join(root, "data", "record", "tips.csv")


# ---------------------------------------------------------------- storage
def _types(df: pd.DataFrame) -> pd.DataFrame:
    for c in _NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)
    return df


class FileStore:
    """Append-only CSV beside the public record."""

    def __init__(self, root: str):
        self.path = path(root)

    def read(self) -> pd.DataFrame:
        if not os.path.isfile(self.path):
            return _types(pd.DataFrame(columns=COLUMNS))
        return _types(pd.read_csv(self.path, dtype=str, keep_default_na=False,
                                  na_values=[""]))

    def append(self, rows: list) -> int:
        if not rows:
            return 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(
            self.path, mode="a", header=not os.path.isfile(self.path),
            index=False)
        return len(rows)

    def describe(self) -> str:
        return "file:%s" % self.path


class DbStore:
    """The same rows in `tip_lists`. The unique key on (day, list, fixture)
    turns two workers freezing the same day into one failed insert."""

    def __init__(self, eng):
        self.engine = eng

    def read(self) -> pd.DataFrame:
        from sqlalchemy import select
        cols = [db.tip_lists.c[c] for c in COLUMNS]
        with self.engine.connect() as c:
            rows = c.execute(
                select(*cols).order_by(db.tip_lists.c.id)).mappings().all()
        return _types(pd.DataFrame([dict(r) for r in rows], columns=COLUMNS))

    def append(self, rows: list) -> int:
        if not rows:
            return 0
        with self.engine.begin() as c:
            c.execute(db.tip_lists.insert(), rows)
        return len(rows)

    def describe(self) -> str:
        return "db:%s" % self.engine.dialect.name


def store(root: str):
    eng = db.engine()
    return DbStore(eng) if eng is not None else FileStore(root)


def load(root: str) -> pd.DataFrame:
    return store(root).read()


def verify(root: str) -> dict:
    return record._check_chain(load(root), COLUMNS)


# ----------------------------------------------------------------- freeze
def _utc(now) -> datetime:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    return now.tz_convert("UTC").to_pydatetime()


def eat_day(now=None) -> str:
    return _utc(now).astimezone(EAT).strftime("%Y-%m-%d")


def eat_time(iso) -> str:
    """'HH:MM' in East Africa Time for a stored UTC timestamp."""
    t = pd.to_datetime(iso, errors="coerce", utc=True)
    return "" if pd.isna(t) else t.tz_convert(EAT).strftime("%H:%M")


def _kickoff(v):
    ko = pd.to_datetime(v, errors="coerce", utc=True)
    return None if pd.isna(ko) else ko


def frozen_days(df: pd.DataFrame) -> set:
    return set(df.loc[df["list"] == DAY_MARK, "day"]) if len(df) else set()


def is_frozen(root: str, now=None) -> bool:
    return eat_day(now) in frozen_days(load(root))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def freeze(slate_rows, root: str, now=None) -> dict:
    """Write today's lists (East Africa Time) once. Returns what was written.

    `slate_rows` are the slate's match dictionaries; only fixtures kicking off
    today, and not yet under way, are eligible - `tips.build` applies the same
    rules as the live Tips screen, so the frozen lists are the ones shown.
    """
    now = _utc(now)
    day = eat_day(now)
    st = store(root)
    have = st.read()
    if day in frozen_days(have):
        return {"frozen": False, "day": day, "written": 0,
                "reason": "already frozen - a day's lists are written once"}

    todays = []
    for m in slate_rows or []:
        ko = _kickoff(m.get("date"))
        if ko is not None and ko.tz_convert(EAT).strftime("%Y-%m-%d") == day:
            todays.append(m)
    out = tips.build(todays, now=now)

    stamp = now.isoformat()
    rows = [{c: None for c in COLUMNS}]
    rows[0].update({"frozen_at": stamp, "day": day, "list": DAY_MARK, "rank": 0,
                    "div": "", "home": "", "away": ""})
    counts = {}
    by_cat = {c["key"]: c["picks"] for c in out.get("categories", [])}
    for name in LISTS:
        picks = out[name] if name in BASE_LISTS else by_cat.get(name, [])
        counts[name] = len(picks)
        for i, c in enumerate(picks, 1):
            ko = _kickoff(c.get("kickoff"))
            rows.append({
                "frozen_at": stamp, "day": day, "list": name, "rank": i,
                "div": c["div"], "league": c.get("league"),
                "date": ko.strftime("%Y-%m-%d") if ko is not None else None,
                "kickoff": ko.isoformat() if ko is not None else None,
                "home": c["home"], "away": c["away"],
                "pick": c["pick"], "side": c["side"],
                "p": _num(c["p"]), "p_dc": _num(c.get("p_double_chance")),
                "market_p": _num(c.get("market_p")), "odds": _num(c.get("odds")),
            })
    prev = str(have["hash"].iloc[-1]) if len(have) else record.GENESIS
    for r in rows:
        r["prev"] = prev
        r["hash"] = record._link(prev, r, COLUMNS)
        prev = r["hash"]
    try:
        st.append(rows)
    except Exception as e:                          # a second worker won the race
        if type(e).__name__ == "IntegrityError":
            return {"frozen": False, "day": day, "written": 0,
                    "reason": "already frozen by another worker"}
        raise
    return {"frozen": True, "day": day, "frozen_at": stamp,
            "written": len(rows) - 1, "counts": counts, "backend": st.describe()}


# ------------------------------------------------------------- settlement
def api_results(root: str) -> pd.DataFrame:
    """Final scores API-Football reported (service/fixtures_api.py)."""
    f = os.path.join(root, "data", "manual", fixtures.API_RESULTS_FILE)
    cols = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Status"]
    if not os.path.isfile(f):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
    for c in ("FTHG", "FTAG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _index(df: pd.DataFrame, with_status: bool) -> dict:
    """(div, home, away) -> [(date, hg, ag, status)] for quick lookup."""
    idx = {}
    if df is None or not len(df):
        return idx
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    for (div, h, a, hg, ag), d, st in zip(
            df[["Div", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].itertuples(
                index=False, name=None),
            dates, df["Status"] if with_status else [None] * len(df)):
        if pd.isna(d):
            continue
        idx.setdefault((str(div), str(h), str(a)), []).append((d, hg, ag, st))
    return idx


def _near(idx, key, day):
    """The entry for this pairing nearest `day`, within a day either side:
    sources date a late kick-off by their own clock, not by UTC."""
    best = None
    for e in idx.get(key, ()):
        gap = abs((e[0] - day).days)
        if gap <= 1 and (best is None or gap < best[0]):
            best = (gap, e)
    return best[1] if best else None


def _played(e) -> bool:
    return e is not None and not pd.isna(e[1]) and not pd.isna(e[2])


def settle(log: pd.DataFrame, official: pd.DataFrame,
           api: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every frozen pick with its outcome, recomputed on each read."""
    picks = log[log["list"] != DAY_MARK].copy() if len(log) else log.copy()
    off_idx = _index(official, with_status=False)
    api_idx = _index(api, with_status=True)
    out = []
    for _, r in picks.iterrows():
        key = (str(r["div"]), str(r["home"]), str(r["away"]))
        day = pd.to_datetime(r["date"], errors="coerce")
        o = a = None
        if not pd.isna(day):
            o = _near(off_idx, key, day.normalize())
            a = _near(api_idx, key, day.normalize())
        a_final = a if (a is not None and a[3] in FINISHED and _played(a)) else None
        hg = ag = None
        source, conflict = None, False
        if _played(o):
            hg, ag, source = int(o[1]), int(o[2]), "official"
            conflict = bool(a_final and (int(a_final[1]), int(a_final[2])) != (hg, ag))
        elif a_final:
            hg, ag, source = int(a_final[1]), int(a_final[2]), "api"
        if hg is not None:
            res = "H" if hg > ag else "A" if hg < ag else "D"
            code = str(r["pick"])
            rule = _SETTLE.get(code)
            if rule is None:
                status, dc = "pending", None       # an unknown code is never a win
            else:
                status = "won" if rule(hg, ag) else "lost"
                # The double chance of a 1X2 pick. Meaningless for a goals
                # line, and left out rather than invented.
                dc = (res in ({"1": "H", "X": "D", "2": "A"}[code], "D")
                      if code in _1X2 else None)
        elif a is not None and a[3] in VOID:
            status, res, dc, source = "void", None, None, "api"
        else:
            status, res, dc = "pending", None, None
        row = r.to_dict()
        row.update({"status": status, "result": res, "dc_won": dc,
                    "score": "%d-%d" % (hg, ag) if hg is not None else None,
                    "source": source, "conflict": conflict})
        out.append(row)
    cols = list(picks.columns) + ["status", "result", "dc_won", "score",
                                  "source", "conflict"]
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------- summary
def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def stats(df: pd.DataFrame) -> dict:
    """Counts and rates for a set of settled-or-not picks."""
    st = df["status"] if len(df) else pd.Series(dtype=str)
    done = df[st.isin(["won", "lost"])] if len(df) else df
    n, won = int(len(done)), int((st == "won").sum())
    out = {"picks": int(len(df)), "settled": n, "won": won, "lost": n - won,
           "void": int((st == "void").sum()), "pending": int((st == "pending").sum()),
           "hit": won / n if n else None,
           "expected": float(done["p"].sum()) if n else None,
           "expected_hit": float(done["p"].mean()) if n else None,
           "dc_hit": float(done["dc_won"].astype(bool).mean()) if n else None,
           "priced": 0, "flat_return": None,
           "enough": n >= tips.MIN_JUDGEABLE}
    if n:
        priced = done[done["odds"].notna()]
        if len(priced):
            gain = np.where(priced["status"] == "won", priced["odds"] - 1.0, -1.0)
            out["priced"] = int(len(priced))
            out["flat_return"] = float(gain.sum() / len(priced))
    return out


def _acca(bankers: pd.DataFrame) -> dict | None:
    """Did the day's bankers all win together? Undecided while any is pending."""
    legs = bankers[bankers["status"] != "void"]
    if not len(legs):
        return None
    if (legs["status"] == "lost").any():
        all_won = False
    elif (legs["status"] == "pending").any():
        all_won = None
    else:
        all_won = True
    return {"legs": int(len(legs)), "all_won": all_won,
            "expected": float(np.prod(legs["p"].astype(float)))}


def _pick(r) -> dict:
    keys = ("rank", "div", "league", "kickoff", "home", "away", "pick", "side",
            "p", "p_dc", "market_p", "odds", "status", "score", "source",
            "conflict", "dc_won")
    return {k: _clean(r[k]) for k in keys}


def summary(root: str, official: pd.DataFrame, api: pd.DataFrame | None = None,
            days: int = 14, now=None) -> dict:
    """Recent frozen days pick by pick, and running totals per list."""
    log = load(root)
    settled = settle(log, official, api)
    today = pd.Timestamp(eat_day(now))
    marks = (log[log["list"] == DAY_MARK].sort_values("day", ascending=False)
             if len(log) else log)

    day_rows = []
    for _, m in marks.head(max(0, int(days))).iterrows():
        sub = settled[settled["day"] == m["day"]] if len(settled) else settled
        lists = {}
        for name in LISTS:
            s = sub[sub["list"] == name].sort_values("rank") if len(sub) else sub
            lists[name] = {"stats": stats(s),
                           "picks": [_pick(r) for _, r in s.iterrows()]}
        bk = sub[sub["list"] == "bankers"] if len(sub) else sub
        day_rows.append({"day": m["day"], "frozen_at": m["frozen_at"],
                         "lists": lists, "acca": _acca(bk) if len(bk) else None})

    totals = {}
    for name in LISTS:
        s = settled[settled["list"] == name] if len(settled) else settled
        dd = pd.to_datetime(s["day"], errors="coerce") if len(s) else None
        totals[name] = {"promised_hit": PROMISED[name]}
        for label, span in (("7", 7), ("30", 30), ("all", None)):
            w = s if span is None or not len(s) else \
                s[dd > today - timedelta(days=span)]
            totals[name][label] = stats(w)
    chain = record._check_chain(log, COLUMNS)
    return {"days": day_rows, "totals": totals, "freeze_hour": FREEZE_HOUR,
            "frozen_days": len(marks), "chain": chain,
            "backend": store(root).describe(),
            "min_judgeable": tips.MIN_JUDGEABLE,
            "conflicts": int(settled["conflict"].sum()) if len(settled) else 0}
