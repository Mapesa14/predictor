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
from service import live

ROOT = os.environ.get("FOOTBALL_DATA", r"D:\Downloads July 2026\SoccerData")
EAT = ZoneInfo("Africa/Dar_es_Salaam")
AFRICAN = {"TZ1", "EG1", "MA1", "DZ1", "ZA1", "NG1",
           "GH1", "KE1", "UG1", "ZM1", "RW1"}

app = FastAPI(title="Football predictor service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_p: Predictor | None = None
_live: live.LiveFetcher | None = None


def predictor() -> Predictor:
    """The engine is stateless in the sense that matters: it re-fits from the
    pool of results on every load, so 'learning' is just newer rows on disk.
    We rebuild lazily whenever the data root changes, which means an offline
    `predict.py update` is picked up by the running service without a restart.
    """
    global _p
    stamp = _data_stamp(ROOT)
    if _p is None or getattr(_p, "_stamp", None) != stamp:
        _p = Predictor(ROOT)
        _p._stamp = stamp
    return _p


def _data_stamp(root: str) -> float:
    newest = 0.0
    for base, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".csv"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(base, fn)))
                except OSError:
                    pass
    return newest


# Optional self-learning loop: with AUTO_REFRESH_HOURS set, the service pulls
# football-data results + the fixtures feed on its own schedule (additive only,
# guarded by the same no-shrink rule as the CLI). Off by default: a network
# fetch inside a server is never a silent default here.
AUTO_REFRESH_HOURS = float(os.environ.get("AUTO_REFRESH_HOURS", "0") or 0)


def _auto_refresh_loop():
    import time
    from predictor import refresh
    while True:
        time.sleep(AUTO_REFRESH_HOURS * 3600)
        try:
            refresh.refresh_all(ROOT, since=time.localtime().tm_year - 1)
            print("auto-refresh complete", flush=True)
        except Exception as e:                       # never take the server down
            print("auto-refresh failed: %r" % e, flush=True)


if AUTO_REFRESH_HOURS > 0:
    import threading
    threading.Thread(target=_auto_refresh_loop, daemon=True).start()
    print("auto-refresh every %.1fh" % AUTO_REFRESH_HOURS, flush=True)


def live_fetcher() -> live.LiveFetcher | None:
    global _live
    key = os.environ.get("LIVE_API_KEY", "").strip()
    if not key:
        return None
    if _live is None:
        ids = [int(x) for x in
               os.environ.get("LIVE_LEAGUES", "").split(",") if x.strip()]
        _live = live.LiveFetcher(key, leagues=ids or None)
    return _live


# ---- team-name matching for the live join ---------------------------------
_ABBREV = {
    "utd": "united", "manu": "manchester", "man": "manchester",
    "spurs": "tottenham", "nott'm": "nottingham", "nottm": "nottingham",
    "wba": "westbromwich", "bha": "brighton",
    "fc": "", "sc": "", "cf": "", "afc": "", "ac": "", "us": "",
    "as": "", "st": "", "ol": "", "om": "",
    "cfc": "", "pdfc": "", "bsc": "", "hsc": "", "ssc": "",
}


def _tokens(name: str) -> frozenset:
    s = "".join(ch if ch.isalnum() else " " for ch in
                name.replace("&", " ")).lower()
    toks = set()
    for w in s.split():
        w = _ABBREV.get(w, w)
        if len(w) >= 4:
            toks.add(w)
    return frozenset(toks)


def match_score(a: str, b: str) -> bool:
    """True for an unambiguous agreement of two distinct names.

    Exact token-set equality, or one set contained in the other with at most
    one extra token ("West Ham United" vs "West Ham"). Short/stop names yield
    no tokens and never match. Used per side of a fixture, so both teams must
    line up.
    """
    if not a or not b:
        return False
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    s, l = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(s) >= 1 and s < l and len(l - s) <= 1


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
    """Real fixtures, then (separately) unpriced head-to-head pairings.

    Rule 2 of UI-PROMPT.md: a fixture must have a published kick-off time.
    Round-robin fallbacks carry a `note` and no time; they are returned in a
    clearly-separated `pairings` surface and are never shown as fixtures.
    """
    p = predictor()
    divs = [league] if league else p.divs
    groups, pair_groups, skipped = [], [], []
    for d in divs:
        if d not in p.divs:
            continue
        sched = p.schedule(d, days)
        if not len(sched):
            continue
        has_note = "note" in sched.columns
        real = sched
        if "Date" in sched.columns:
            real = real[real["Date"].notna()]
        if "Time" in sched.columns:
            real = real[real["Time"].notna()]
        if has_note:
            real = real[real["note"].isna()]
        pairings = sched[~sched.index.isin(real.index)]
        matches = []
        for _, f in real.iterrows():
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
        if len(pairings):
            pair_rows = []
            for _, f in pairings.iterrows():
                try:
                    s = p.predict(f["HomeTeam"], f["AwayTeam"], d,
                                  allow_new=True)
                except SystemExit:
                    continue
                r = s["result"]
                pick = max(r, key=r.get)
                pair_rows.append({
                    "home": s["home"], "away": s["away"],
                    "pick": {"H": "1", "D": "X", "A": "2"}[pick],
                    "p": {"1": r["H"], "X": r["D"], "2": r["A"]},
                    "note": (f.get("note") or "") if has_note else "",
                })
            if pair_rows:
                pair_groups.append({
                    "code": d, "league": leagues.name(d),
                    "country": leagues.country(d),
                    "matches": pair_rows,
                })
    groups.sort(key=lambda g: _group_order(g["code"]))
    pair_groups.sort(key=lambda g: _group_order(g["code"]))
    return {"generated": datetime.now(EAT).isoformat(),
            "groups": groups, "pairings": pair_groups, "skipped": skipped}


@app.get("/api/live")
def api_live(date: str | None = None):
    """Latest live/FT scores from the provider. Empty when no key is set.

    Joins against nothing here: the PWA keys the rows by (home, away) after
    normalising names, so a mismatch simply means no live chip, never a wrong
    score on a fixture.
    """
    d = date or datetime.now(EAT).strftime("%Y-%m-%d")
    fetcher = live_fetcher()
    if fetcher is None:
        return {"provider": "api-football", "enabled": False,
                "date": d, "matches": [],
                "note": "set LIVE_API_KEY to enable live scores"}
    matches = fetcher.fetch_date(d)
    return {"provider": "api-football", "enabled": True,
            "date": d, "matches": matches,
            "leagues": len(fetcher.leagues)}


@app.get("/api/live/leagues")
def api_live_leagues(search: str):
    """Provider-side league search, to fix DIV_LEAGUES / LIVE_LEAGUES ids."""
    fetcher = live_fetcher()
    if fetcher is None:
        return {"enabled": False,
                "note": "set LIVE_API_KEY to enable live scores"}
    return {"enabled": True, "results": live.find_leagues(search, fetcher.key)}


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