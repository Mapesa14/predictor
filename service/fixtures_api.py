"""Fixtures from API-Football, for every league and cup the feed has not published.

football-data.co.uk's fixtures feed is the product's European schedule, and on
Tuesday 15 September 2026 it still stopped at the Sunday before. That day there
were 31 fixtures in leagues the product carries - La Liga, the Championship,
the Scottish Premiership, the Eredivisie - plus League Cup and Coppa Italia
ties, and the app showed exactly one of them. Egypt, Uganda and Rwanda had no
fixtures source at all. API-Football lists every fixture on a date in a single
request, so this fills the gap for two requests per refresh.

Three rules, because the failure that matters is a prediction for the wrong
club, not a missing one:

  * Names are matched strictly. The provider spells clubs its own way
    ("Espanyol", "Heart Of Midlothian", "Sutton Utd"); our results use
    football-data's ("Espanol", "Hearts", "Sutton"). A provider name is taken
    only through an explicit alias, the engine's own exact lookup, or by
    dropping a generic word ("Town", "FC", "Utd") to land on exactly one club.
    Anything else is reported and left out - never guessed.
  * A league whose results in our data are stale is not priced from these
    fixtures. Egypt's stop at May 2025.
  * Published sources win. A fixture the football-data feed or a league's own
    site also lists is taken from there - they carry closing prices and the
    league's own kick-off times. These rows only fill gaps (see
    fixtures.load_any).

Cups are the same list with a competition name. A tie between two clubs in one
division is priced on that division. A tie across divisions is priced on the
higher division's scale where a gap between the two has been measured
(England's ladder) with the lower division recorded in ScaleAlt, because the
two scales disagree and the method is untested on cup results. Anything else is
reported as unpriced.

Every request goes through the live-score budget ledger in service/live.py.
"""
from __future__ import annotations

import os
import re
import unicodedata
import urllib.parse
from datetime import timedelta

import pandas as pd

from predictor import fixtures, leagues
from service import live

COLUMNS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "Comp", "ScaleAlt"]

# A league whose latest result in our data is older than this is not priced
# from API fixtures. Long enough to cover a European summer break; short enough
# to catch a league whose data stopped a season ago.
STALE_DAYS = 150

# Domestic cups in the countries we carry, ids from the provider's own list.
CUPS = {
    45: ("FA Cup", "England"), 48: ("League Cup", "England"),
    143: ("Copa del Rey", "Spain"), 137: ("Coppa Italia", "Italy"),
    81: ("DFB Pokal", "Germany"), 66: ("Coupe de France", "France"),
    90: ("KNVB Beker", "Netherlands"), 96: ("Taça de Portugal", "Portugal"),
    97: ("Taça da Liga", "Portugal"), 147: ("Belgian Cup", "Belgium"),
    206: ("Türkiye Kupası", "Turkey"), 199: ("Greek Cup", "Greece"),
    181: ("Scottish Cup", "Scotland"), 185: ("Scottish League Cup", "Scotland"),
}

# Provider spelling -> the name our results use, per country. Every entry is a
# club checked by hand against its own record; add to it from the unresolved
# report `predict.py refresh-fixtures-api` prints, and only when sure.
ALIASES = {
    "England": {
        "Boston United": "Boston Utd", "Solihull Moors": "Solihull",
        "Peterborough": "Peterboro", "Manchester City": "Man City",
        "Manchester United": "Man United", "Sheffield Utd": "Sheffield United",
        "Nottingham Forest": "Nott'm Forest",
        "Sheffield Wednesday": "Sheffield Weds",
        "Kidderminster Harriers": "Kidderminster",
    },
    "Spain": {
        "Rayo Vallecano": "Vallecano", "Espanyol": "Espanol",
        "Deportivo La Coruna": "La Coruna", "Racing Santander": "Santander",
        "Athletic Club": "Ath Bilbao", "Atletico Madrid": "Ath Madrid",
        "Real Betis": "Betis", "Celta Vigo": "Celta", "Real Sociedad": "Sociedad",
    },
    "Scotland": {"Heart Of Midlothian": "Hearts"},
    # Niki Volos is not Volos NFC - a different club, deliberately left out.
    "Greece": {"Levadiakos": "Levadeiakos", "Aris Thessalonikis": "Aris"},
    # The two Sudanese clubs playing the Rwandan league as guests.
    "Rwanda": {"Al Merreikh": "Al-Merreikh (Omdurman)",
               "Al Hilal Omdurman": "Al-Hilal (Omdurman)"},
}

# Generic words dropped from a provider name, to find the one club it leaves.
_GENERIC = {"fc", "afc", "cf", "sc", "town", "utd", "united", "city", "club",
            "cd", "ud", "sd", "ac", "as", "calcio"}

_SKIP_STATUS = {"PST", "CANC", "ABD", "AWD", "WO"}   # postponed, cancelled, ...


def api_file(root: str) -> str:
    return os.path.join(root, "data", "manual", fixtures.API_FILE)


# ------------------------------------------------------------------ names
def _fold(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _strip_generic(s) -> str:
    return " ".join(w for w in _fold(s).split() if w not in _GENERIC)


def _one_club(hits, divs):
    """The club if every hit names the same one, preferring the earlier
    division in `divs` (the fixture's own); None if two different clubs."""
    names = {t for t, _ in hits}
    if len(names) != 1:
        return None
    team = names.pop()
    for d in divs:
        if (team, d) in hits:
            return team, d
    return None


def resolve_club(p, name: str, country: str, divs: list):
    """(team, division) for a provider club name, or None. Never a guess."""
    divs = [d for d in divs if d in p.divs]
    if not divs:
        return None
    target = ALIASES.get(country, {}).get(name)
    if target:
        return _one_club({(t, d) for d in divs for t in p.teams(d) if t == target},
                         divs)
    hits = set()
    for d in divs:
        try:
            hits.add(p.resolve(name, d, fuzzy=False))
        except SystemExit:
            pass
    got = _one_club(hits, divs)
    if got:
        return got
    base = _strip_generic(name)
    if base and base != _fold(name):
        return _one_club({(t, d) for d in divs for t in p.teams(d)
                          if _fold(t) == base}, divs)
    return None


def _scope(div: str) -> list:
    """The fixture's division first, then its ladder neighbours by distance -
    a club promoted or relegated since last season is still found."""
    lad = leagues.ladder_of(div)
    if not lad:
        return [div]
    return sorted(lad, key=lambda o: abs(lad.index(o) - lad.index(div)))


def _country_divs(p, country: str) -> list:
    divs = [d for d in p.divs if leagues.country(d) == country]
    lad = leagues.ladder_of(divs[0]) if divs else None
    if lad:
        return sorted(divs, key=lambda d: lad.index(d) if d in lad else len(lad))
    return sorted(divs)


# ------------------------------------------------------------------- rows
def _stale(last, div, when, stale_days) -> bool:
    if stale_days is None:
        return False
    d = last.get(div)
    if d is None or pd.isna(d):
        return True
    return (when.tz_convert(None) - pd.Timestamp(d)).days > stale_days


def _row(div, when, home, away, comp="", alt=""):
    """Date and time in the division's own source zone, the rule every reader
    of a fixtures file already applies (leagues.kickoff)."""
    local = when.tz_convert(leagues.source_tz(div))
    return {"Div": div, "Date": local.strftime("%d/%m/%Y"),
            "Time": local.strftime("%H:%M"), "HomeTeam": home, "AwayTeam": away,
            "Comp": comp, "ScaleAlt": alt}


def build_rows(provider_fixtures, p, stale_days=STALE_DAYS):
    """Provider fixtures -> fixtures-file rows, and a report of what was left out."""
    ours = {v: k for k, v in live.DIV_LEAGUES.items()}
    last = p.df.groupby("Div")["Date"].max()
    rows = []
    report = {"unresolved": [], "not_loaded": [], "stale": [], "no_gap": []}
    for f in provider_fixtures or []:
        try:
            lg = f.get("league") or {}
            lid = lg.get("id")
            home = f["teams"]["home"]["name"]
            away = f["teams"]["away"]["name"]
            when = pd.Timestamp(f["fixture"]["date"])
        except (KeyError, TypeError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        status = ((f.get("fixture") or {}).get("status") or {}).get("short")
        if status in _SKIP_STATUS:
            continue
        label = "%s v %s" % (home, away)

        if lid in ours:
            div = ours[lid]
            if div not in p.divs:
                report["not_loaded"].append("%s (%s)" % (label, lg.get("name")))
                continue
            if _stale(last, div, when, stale_days):
                report["stale"].append("%s (%s)" % (label, leagues.name(div)))
                continue
            scope, country = _scope(div), leagues.country(div)
            h = resolve_club(p, home, country, scope)
            a = resolve_club(p, away, country, scope)
            missing = [n for n, r in ((home, h), (away, a)) if r is None]
            if missing:
                report["unresolved"].append("%s (%s): %s" % (
                    label, leagues.name(div), ", ".join(missing)))
                continue
            rows.append(_row(div, when, h[0], a[0]))

        elif lid in CUPS:
            comp, country = CUPS[lid]
            scope = _country_divs(p, country)
            if not scope:
                continue                     # a cup in a country we do not carry
            h = resolve_club(p, home, country, scope)
            a = resolve_club(p, away, country, scope)
            missing = [n for n, r in ((home, h), (away, a)) if r is None]
            if missing:
                report["unresolved"].append("%s (%s): %s" % (
                    label, comp, ", ".join(missing)))
                continue
            if h[1] == a[1]:
                div, alt = h[1], ""
            else:
                lad = leagues.ladder_of(h[1])
                if lad is None or a[1] not in lad:
                    report["no_gap"].append("%s (%s): %s v %s" % (
                        label, comp, leagues.name(h[1]), leagues.name(a[1])))
                    continue
                div = min(h[1], a[1], key=lad.index)    # the higher division
                alt = max(h[1], a[1], key=lad.index)
            if _stale(last, div, when, stale_days):
                report["stale"].append("%s (%s)" % (label, comp))
                continue
            rows.append(_row(div, when, h[0], a[0], comp, alt))
    return rows, report


# ------------------------------------------------------------------ fetch
def fetch_day(day: str, store=None, now=None, transport=None) -> list:
    """Every fixture on one date (UTC), through the budget ledger."""
    if not live.configured():
        raise RuntimeError("LIVE_API_KEY is not set - nothing was sent")
    store = store or live.default_store()
    now = live._naive_utc(now) or live.utcnow()
    b = live.budget(store, now)
    if b["cooling_down_until"]:
        raise RuntimeError("provider rate limit; cooling down until %s"
                           % b["cooling_down_until"])
    if b["available"] <= 0:
        raise RuntimeError("request budget spent for the rolling 24 hours")
    if b["used_last_minute"] >= max(1, live.MINUTE_LIMIT - 1):
        raise RuntimeError("per-minute limit reached")
    endpoint = "/fixtures?" + urllib.parse.urlencode({"date": day, "timezone": "UTC"})
    call_id = store.begin_call(endpoint, now)
    try:
        status, headers, raw = (transport or live.http_get)(
            live.BASE + endpoint, live._headers(), 30)
    except Exception as e:
        store.finish_call(call_id, None, False, None, None,
                          ("unsent: " if live._never_sent(e) else "network: ")
                          + live._describe(e))
        raise RuntimeError("fixtures request failed: %s" % live._describe(e))
    hdr = {str(k).lower(): v for k, v in (headers or {}).items()}
    body = live._json(raw)
    errors = body.get("errors") if isinstance(body, dict) else None
    ok = status == 200 and not errors
    rows = (body.get("response") or []) if ok else []
    if status == 429 or live._mentions_limit(errors):
        note = "rate-limited: %s" % (live._short(errors) or "HTTP %s" % status)
    elif ok:
        note = "ok: %d fixtures on %s" % (len(rows), day)
    else:
        note = "error: %s" % (live._short(errors) or "HTTP %s" % status)
    store.finish_call(call_id, status, ok,
                      live._to_int(hdr.get("x-ratelimit-requests-remaining")),
                      live._to_int(hdr.get("x-ratelimit-remaining")), note)
    if not ok:
        raise RuntimeError("fixtures request refused: %s" % note)
    return rows


def _iso(ddmmyyyy: str) -> str:
    try:
        return pd.to_datetime(ddmmyyyy, format="%d/%m/%Y").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def sync(root: str, p, days: int = 2, now=None, store=None, transport=None,
         stale_days=STALE_DAYS) -> dict:
    """Fetch today and the next days, and rewrite the API fixtures file.

    A day that could not be fetched keeps whatever the file already had for it,
    so a failed request never empties the schedule.
    """
    now = live._naive_utc(now) or live.utcnow()
    dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(max(1, int(days)))]
    fetched, errors, rows = [], [], []
    report = {"unresolved": [], "not_loaded": [], "stale": [], "no_gap": []}
    for day in dates:
        try:
            got = fetch_day(day, store=store, now=now, transport=transport)
        except RuntimeError as e:
            errors.append("%s: %s" % (day, e))
            continue
        fetched.append(day)
        r, rep = build_rows(got, p, stale_days=stale_days)
        rows += r
        for k, v in rep.items():
            report[k].extend(v)

    path = api_file(root)
    out = pd.DataFrame(rows, columns=COLUMNS)
    if fetched:
        if os.path.isfile(path):
            old = pd.read_csv(path, dtype=str).fillna("")
            failed = set(dates) - set(fetched)
            keep = old[old["Date"].map(_iso).isin(failed)] if failed else old.iloc[0:0]
            out = pd.concat([keep.reindex(columns=COLUMNS), out], ignore_index=True)
        out = out.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"],
                                  keep="last")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        out.to_csv(tmp, index=False)
        os.replace(tmp, path)
    cups = int((out["Comp"].fillna("") != "").sum()) if len(out) else 0
    return {"fetched": fetched, "errors": errors, "rows": int(len(out)),
            "leagues": int(len(out)) - cups, "cups": cups, "report": report,
            "file": path, "written": bool(fetched)}
