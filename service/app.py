"""The engine behind the PWA: FastAPI model service.

Every card is one score matrix, so every market on a card is a slice of the
same probabilities — the front end never re-fits anything.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predictor import leagues
from predictor.engine import Predictor

ROOT = os.environ.get("FOOTBALL_DATA", r"D:\Downloads July 2026\SoccerData")
EAT = ZoneInfo("Africa/Dar_es_Salaam")
AFRICAN = {"TZ1", "EG1", "MA1", "DZ1", "ZA1", "NG1",
           "GH1", "KE1", "UG1", "ZM1", "RW1"}

app = FastAPI(title="Football predictor service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_p: Predictor | None = None


def predictor() -> Predictor:
    global _p
    if _p is None:
        _p = Predictor(ROOT)
    return _p


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float):
        return round(v, 4)
    return v


def _local_kickoff(f) -> tuple[datetime | None, str]:
    """Kick-off in East Africa Time, from the feed's UK date (and time if set)."""
    if pd.isna(f.get("Date")):
        return None, ""
    d = f["Date"]
    if f.get("Time"):
        try:
            t = pd.to_datetime(f["Time"], format="%H:%M").time()
        except (TypeError, ValueError):
            t = None
    else:
        t = None
    if t is None:
        return None, d.strftime("%a %d %b")
    uk = ZoneInfo("Europe/London")
    dt = datetime.combine(d, t, tzinfo=uk).astimezone(EAT)
    return dt, dt.strftime("%a %d %b %H:%M")


@app.get("/health")
def health():
    return {"ok": True}


def _group_order(code: str) -> tuple:
    tz = code == "TZ1"                      # Tanzania pinned above Europe
    fed = code in AFRICAN                   # other African federations next
    top = code in leagues.TOP_10
    return (0 if tz else 1 if fed else 2 if top else 3,
            leagues.name(code).casefold())


@app.get("/api/slate")
def slate(days: int = 2, league: str | None = None):
    p = predictor()
    divs = [league] if league else p.divs
    groups, skipped = [], []
    for d in divs:
        if d not in p.divs:
            continue
        sched = p.schedule(d, days)
        if not len(sched):
            continue
        matches = []
        for _, f in sched.iterrows():
            odds = f if pd.notna(f.get("AvgH")) else None
            try:
                s = p.predict(f["HomeTeam"], f["AwayTeam"], d,
                              allow_new=True, odds=odds)
            except SystemExit as e:
                skipped.append("%s: %s v %s (%s)"
                               % (leagues.name(d), f["HomeTeam"],
                                  f["AwayTeam"], e))
                continue
            i, j, _pct = s["correct_scores"][0]
            r = s["result"]
            pick = max(r, key=r.get)
            ko, label = _local_kickoff(f)
            started = bool(ko and ko <= datetime.now(EAT))
            matches.append({
                "div": d,
                "league": leagues.name(d),
                "date": (ko or f["Date"]).isoformat(),
                "kickoff_label": label,
                "home": s["home"], "away": s["away"],
                "pick": {"H": "1", "D": "X", "A": "2"}[pick],
                "p": {"1": r["H"], "X": r["D"], "2": r["A"]},
                "o25": s["totals"][2.5]["over"],
                "btts": s["btts"]["yes"],
                "score": "%d-%d" % (i, j),
                "xg": "%.2f-%.2f" % (s["exp_home"], s["exp_away"]),
                "new": bool(s["home_new"] or s["away_new"]),
                "started": started,
            })
        if matches:
            groups.append({
                "code": d, "league": leagues.name(d),
                "country": leagues.country(d),
                "matches": sorted(matches, key=lambda m: m["date"]),
            })
    groups.sort(key=lambda g: _group_order(g["code"]))
    return {"generated": datetime.now(EAT).isoformat(),
            "groups": groups, "skipped": skipped}


@app.get("/api/card")
def card(home: str, away: str, div: str | None = None, neutral: bool = False):
    s = predictor().predict(home, away, div, neutral=neutral, allow_new=bool(div))
    return _jsonable(s)


@app.get("/api/leagues")
def leagues_list():
    p = predictor()
    out = []
    for d in p.divs:
        df = p.df[p.df["Div"] == d]
        out.append({
            "code": d, "league": leagues.name(d), "country": leagues.country(d),
            "matches": int(len(df)),
            "covered": [df["Date"].min().strftime("%Y-%m-%d"),
                        df["Date"].max().strftime("%Y-%m-%d")],
            "top10": d in leagues.TOP_10,
        })
    return out


@app.get("/api/table")
def table(div: str, top: int = 8):
    p = predictor()
    fm = p.models(div)["FT"]
    rows = []
    for t, att, deff, rating, played in model_strength(fm):
        rows.append({"team": t, "attack": round(att, 3),
                     "defence": round(deff, 3),
                     "rating": round(rating, 3), "played": played})
    return {"div": div, "league": leagues.name(div),
            "rows": rows[:top or None], "source_notes": True}


def model_strength(fm):
    """(team, attack, defence, rating, played) rows like the table command."""
    if not hasattr(fm, "attack"):
        return []
    avgd = fm.mean_defence()
    out = []
    for t, att in sorted(fm.attack.items(),
                         key=lambda kv: kv[1] / fm.defence.get(kv[0], 1),
                         reverse=True):
        deff = fm.defence.get(t, avgd)
        out.append((t, att, deff, att / deff, fm.played.get(t, 0)))
    return out