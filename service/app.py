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

from predictor import envfile

# Before anything reads the environment: a local .env supplies LIVE_API_KEY and
# friends when running without docker compose. Real variables still win.
envfile.load()

from predictor import db, leagues, market, record, tips  # noqa: E402
from predictor.engine import Predictor  # noqa: E402
from service import live  # noqa: E402

ROOT = os.environ.get("FOOTBALL_DATA", r"D:\Downloads July 2026\SoccerData")
EAT = ZoneInfo("Africa/Dar_es_Salaam")
AFRICAN = {"TZ1", "EG1", "MA1", "DZ1", "ZA1", "NG1",
           "GH1", "KE1", "UG1", "ZM1", "RW1"}

app = FastAPI(title="Football predictor service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_p: Predictor | None = None
_warm = {"done": False, "at": None}


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


# ---- warm-up ---------------------------------------------------------------
# Fitting every division on the first request made the slate a ~45s cold start.
# Fit all divisions in the background at boot, then precompute the default
# slates so the first real request is served from the cache.
import threading as _t
import time as _time

_slate_cache: dict = {}
_SLATE_TTL = 60.0


def _warmup():
    try:
        if db.url():
            # create_all creates what is missing and leaves existing tables
            # alone, so running it on every boot is safe.
            try:
                db.create_all()
                print("database: tables ready (%s)" % db.engine().dialect.name,
                      flush=True)
            except Exception as e:
                print("database: could not initialise: %r" % e, flush=True)
        p = predictor()
        for d in p.divs:
            p.models(d)
        print("warm-up: %d divisions fitted" % len(p.divs), flush=True)
        for days in (1, 2):
            try:
                _cached_slate(days, None)
            except Exception as e:
                print("warm-up slate %d failed: %r" % (days, e), flush=True)
        print("warm-up: default slates cached", flush=True)
        _warm["done"] = True
        _warm["at"] = datetime.now(EAT).isoformat()
        # Live scores start only once the slate exists, because the slate's
        # kick-off times are what decide whether a provider call is worth it.
        if live.configured():
            live.start_refresher(_kickoffs_today)
            print("live scores: refresher started", flush=True)
    except Exception as e:
        print("warm-up failed: %r" % e, flush=True)


_t.Thread(target=_warmup, daemon=True).start()


def _kickoffs_today() -> list:
    """Every kick-off on today's slate, for the live refresher's in-play test.

    Read from the cached slate rather than recomputed, so asking every minute
    is free. A fixture with no published time has no kick-off and cannot put
    the refresher into play.
    """
    try:
        slate = _cached_slate(1, None)
    except Exception:
        return []
    out = []
    for g in slate.get("groups", []):
        for m in g.get("matches", []):
            ts = pd.to_datetime(m.get("date"), errors="coerce", utc=True)
            if not pd.isna(ts):
                out.append(ts.to_pydatetime())
    return out


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
        out = {}
        for k, x in v.items():
            key = k if not isinstance(k, tuple) else "/".join(str(t) for t in k)
            out[key] = _jsonable(x)
        return out
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
    """Kick-off in East Africa Time, and its label. See `leagues.kickoff`.

    The per-competition source zone lives in the library, not here, so the CLI
    that writes the public record converts identically.
    """
    if pd.isna(f.get("Date")):
        return None, ""
    dt = leagues.kickoff(f.get("Div"), f["Date"], f.get("Time"))
    if dt is None:
        return None, pd.Timestamp(f["Date"]).strftime("%a %d %b")
    return dt, dt.strftime("%a %d %b %H:%M")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/health")
def api_health():
    """Liveness plus what a deploy actually needs to know.

    Under /api/ so it answers through the same proxy as everything else - the
    nginx config forwards only /api/, which left the bare /health unreachable
    from outside - and so the container healthcheck has a real route to probe.
    Always 200 while the process is up: a slow warm-up or a missing key is
    reported rather than treated as down, or an orchestrator would restart the
    service in a loop during the eleven-second model fit.
    """
    return {"ok": True, "warm": _warm["done"], "warm_at": _warm["at"],
            "data": _data_status(),
            "database": db.healthy(), "live": live.status()}


def _data_status() -> dict | None:
    """Which match data actually loaded, so a half-mounted deploy is visible.

    The loader never fails when the European pool is missing: it merges the
    bundled African and Nordic divisions with an empty mount, serves fifteen
    divisions instead of thirty, and every endpoint answers 200. That is the
    failure this exists to surface. It reads the engine already loaded and
    never triggers a load, so a health check during warm-up stays instant.
    """
    p = _p
    if p is None:
        return None
    have = set(p.df["Div"].unique())
    return {"root": ROOT, "divisions": len(have), "matches": int(len(p.df)),
            "missing_top10": [d for d in leagues.TOP_10 if d not in have],
            "fixtures_feed": os.path.exists(os.path.join(ROOT, "fixtures.csv"))}


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
    Cached per data stamp, so repeated hits are milliseconds.
    """
    return _cached_slate(days, league)


def _cached_slate(days: int, league: str | None):
    stamp = _data_stamp(ROOT)
    now = _time.time()
    hit = _slate_cache.get((days, league))
    if hit and hit["stamp"] == stamp and (now - hit["at"]) < _SLATE_TTL:
        return hit["payload"]
    payload = _compute_slate(days, league)
    _slate_cache[(days, league)] = {"stamp": stamp, "at": now, "payload": payload}
    return payload


def _compute_slate(days: int, league: str | None):
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
            mk = None
            if odds is not None:
                try:
                    # Shin - the same de-vig the engine blends with and the
                    # backtest benchmarks against. Normalising 1/odds instead
                    # is a different quantity, up to 2.9 points apart on a
                    # lopsided market, and the comparison surface exists
                    # precisely to show gaps of about that size.
                    q = market.devig([float(f["AvgH"]), float(f["AvgD"]),
                                      float(f["AvgA"])], "shin")
                    mk = {"1": float(q[0]), "X": float(q[1]), "2": float(q[2])}
                except (ValueError, TypeError, KeyError):
                    # Only bad prices are tolerated here. A blanket `except`
                    # once swallowed a NameError and silently emptied the
                    # market column on every fixture.
                    mk = None
            matches.append({
                "div": d,
                "league": leagues.name(d),
                "date": (ko or f["Date"]).isoformat(),
                "kickoff_label": label,
                "home": s["home"], "away": s["away"],
                "pick": {"H": "1", "D": "X", "A": "2"}[pick],
                "p": {"1": r["H"], "X": r["D"], "2": r["A"]},
                "market": mk,
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
        # Pairings are counted here but priced only when a reader opens one.
        # Pricing all of them inline meant ~2,000 fits and a 42-second, 324KB
        # response for a screen whose real content is a few dozen fixtures.
        if len(pairings):
            pair_groups.append({
                "code": d, "league": leagues.name(d),
                "country": leagues.country(d),
                "count": int(len(pairings)),
                "note": next((n for n in pairings.get("note", pd.Series(dtype=str))
                              if isinstance(n, str) and n), "") if has_note else "",
            })
    groups.sort(key=lambda g: _group_order(g["code"]))
    pair_groups.sort(key=lambda g: _group_order(g["code"]))
    return {"generated": datetime.now(EAT).isoformat(),
            "groups": groups, "pairings": pair_groups, "skipped": skipped}


# ---- the public record ------------------------------------------------------
# Where the record lives. Deliberately not ROOT: the results pool is a data
# directory that gets re-downloaded and swapped, and the record has to outlive
# that. RECORD_ROOT overrides it, which is what a deployment wants - the record
# is the one piece of state that must sit on a persistent volume and survive
# every redeploy, because it cannot be regenerated.
REPO = os.environ.get("RECORD_ROOT") or \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.post("/api/record/publish")
def record_publish(days: int = 2):
    """Freeze today's slate into the record. Only fixtures still to kick off.

    Safe to call repeatedly - a fixture already recorded is skipped, and one
    that has started is refused. Meant for a scheduler a few times a day.
    """
    slate = _cached_slate(days, None)
    rows = [m for g in slate["groups"] for m in g["matches"]]
    return record.publish(rows, REPO)


@app.get("/api/record")
def record_summary():
    """What was predicted, what happened, and whether the file was touched."""
    return _jsonable(record.summary(REPO, predictor().df))


@app.get("/api/tips")
def api_tips(days: int = 2):
    """Clear picks from the slate: a short list, a long list, and what to avoid.

    Built from the same cached slate the fixtures screen uses, so a tip can
    never disagree with its own card. Carries the measurement behind the rules
    and how those rules have done on the published record, because a list of
    "strong" picks with no evidence attached is exactly the tipster page this
    product exists not to be. See predictor/tips.py.
    """
    slate = _cached_slate(days, None)
    rows = [m for g in slate.get("groups", []) for m in g.get("matches", [])]
    out = tips.build(rows)
    try:
        settled = record.settle(record.load(REPO), predictor().df)
        out["record"] = tips.record_performance(settled)
    except Exception as e:
        out["record"] = {"error": "%s: %s" % (type(e).__name__, e)}
    out["days"] = days
    out["generated"] = slate.get("generated")
    return _jsonable(out)


@app.get("/api/record/verify")
def record_verify():
    return record.verify(REPO)


@app.get("/api/pairings")
def pairings(div: str, days: int = 2, limit: int = 60):
    """Head-to-head ratings for one division, priced on demand.

    These are not fixtures: they are the pairings a league has not played yet,
    with no published kick-off. Kept off the slate so opening the app does not
    pay for two thousand predictions nobody asked for.
    """
    p = predictor()
    if div not in p.divs:
        return {"code": div, "league": leagues.name(div), "matches": [],
                "note": "no data loaded for %s" % div}
    sched = p.schedule(div, days)
    if not len(sched):
        return {"code": div, "league": leagues.name(div), "matches": [], "note": ""}
    has_note = "note" in sched.columns
    real = sched
    if "Date" in sched.columns:
        real = real[real["Date"].notna()]
    if "Time" in sched.columns:
        real = real[real["Time"].notna()]
    if has_note:
        real = real[real["note"].isna()]
    rows = sched[~sched.index.isin(real.index)]
    out = []
    for _, f in rows.head(max(1, limit)).iterrows():
        try:
            s = p.predict(f["HomeTeam"], f["AwayTeam"], div, allow_new=True)
        except SystemExit:
            continue
        r = s["result"]
        pick = max(r, key=r.get)
        out.append({
            "home": s["home"], "away": s["away"],
            "pick": {"H": "1", "D": "X", "A": "2"}[pick],
            "p": {"1": r["H"], "X": r["D"], "2": r["A"]},
            "note": (f.get("note") or "") if has_note else "",
        })
    return {"code": div, "league": leagues.name(div),
            "country": leagues.country(div), "matches": out,
            "total": int(len(rows)), "shown": len(out)}


@app.get("/api/live")
def api_live():
    """The latest live scores - read from the database, never from the provider.

    Every open client polls this once a minute. If it could trigger a provider
    call, a few open tabs would spend API-Football's free day in minutes - and
    the version this replaces did exactly that, 38 leagues one request each, so
    one cold page load was nearly four times the 10-a-minute cap. It now only
    reads the last snapshot; one background refresher decides when the
    provider is worth a request. See service/live.py.

    The PWA still matches rows on both team names, so a mismatch means no
    live chip, never a score on the wrong fixture. League search, which used
    to be a public route here that any visitor could spend quota with, is now
    the admin-only `predict.py live-leagues`.
    """
    return live.read()


@app.get("/api/card")
def card(home: str, away: str, div: str | None = None, neutral: bool = False):
    s = predictor().predict(home, away, div, neutral=neutral, allow_new=bool(div))
    return _jsonable(s)


@app.get("/api/clubs")
def clubs(div: str):
    """Rating table for one division, strongest first."""
    p = predictor()
    try:
        rows = p.clubs(div)
    except SystemExit as e:
        return {"code": div, "league": leagues.name(div), "clubs": [],
                "note": "can't fit %s yet: %s" % (div, e)}
    return {"code": div, "league": leagues.name(div), "clubs": rows}


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