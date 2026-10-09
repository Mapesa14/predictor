"""Whether every source is still arriving - and saying so on screen.

The failure this exists to stop happened on 2026-10-09: the fixtures feed had
not been downloaded since 12 September and the Tanzanian refresh had been
crashing for three weeks, and the product said nothing at all. It showed an
empty day, in the same confident typeface as a full one. Silence is the worst
possible answer, because a reader cannot tell "no matches today" from "this
app stopped working three weeks ago".

Two things are measured, and they are not the same question:

  * **when a source last arrived** - the file's own timestamp. A league can go
    a fortnight without a match (an international break) while the pipeline is
    perfectly healthy, so staleness is judged on the refresh, never on the last
    result.
  * **what the schedule actually covers** - whether there are fixtures for
    today and tomorrow at all. That is the question a reader is really asking,
    and the one that was answered wrongly.

Levels are `ok`, `warn` and `stale`. `warn` means a refresh has been missed;
`stale` means nobody can rely on what is on screen.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import fixtures

EAT = ZoneInfo("Africa/Dar_es_Salaam")

# Hours since a source last refreshed before it is a warning, then stale. The
# deployed service refreshes every 6 hours, so a day's silence is already a
# missed cycle; three days means nobody is watching.
AGE_WARN = 26.0
AGE_STALE = 72.0
API_AGE_WARN = 12.0          # fetched every refresh cycle, so sooner
API_AGE_STALE = 36.0

_LEVELS = {"ok": 0, "warn": 1, "stale": 2}

# Two different audiences. "data" is what the reader is looking at - a stale
# source there means the numbers on screen are wrong, and the screen has to say
# so. "publishing" is the product keeping its own promises (the record, the
# frozen tip lists); behind there is an operational fault, not a reason to
# distrust today's predictions, and saying "this data is out of date" over a
# perfectly current slate would train readers to ignore the warning.
SCOPES = {"schedule": "data", "feed": "data", "api_fixtures": "data",
          "tanzania": "data", "api_results": "data", "results": "data",
          "record": "publishing", "tips": "publishing"}


def _worst(levels) -> str:
    return max(levels, key=lambda l: _LEVELS.get(l, 0), default="ok")


def _now(now=None) -> datetime:
    n = pd.Timestamp(now or datetime.now(timezone.utc))
    if n.tzinfo is None:
        n = n.tz_localize("UTC")
    return n.tz_convert("UTC").to_pydatetime()


def _age_hours(path: str, now: datetime):
    if not path or not os.path.exists(path):
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    return (now - mtime).total_seconds() / 3600.0


def _by_age(age, warn=AGE_WARN, stale=AGE_STALE) -> str:
    if age is None:
        return "stale"
    return "stale" if age > stale else "warn" if age > warn else "ok"


def _said(age) -> str:
    if age is None:
        return "never"
    if age < 1:
        return "%d min ago" % max(1, round(age * 60))
    if age < 48:
        return "%d hours ago" % round(age)
    return "%d days ago" % round(age / 24)


def _newest(paths) -> str | None:
    found = [p for p in paths if os.path.exists(p)]
    return max(found, key=os.path.getmtime) if found else None


def _source(key, label, path, now, warn=AGE_WARN, stale=AGE_STALE, fix="",
            note="") -> dict:
    age = _age_hours(path, now)
    return {"key": key, "label": label, "level": _by_age(age, warn, stale),
            "age_hours": None if age is None else round(age, 1),
            "last": _said(age), "note": note, "fix": fix, "path": path}


def schedule_cover(data_root: str, here: str, now=None) -> dict:
    """Fixtures on the board for today and tomorrow, East Africa Time.

    No fixtures tomorrow is the symptom a reader sees first, and it is reported
    as a fault rather than an empty page: a normal day has dozens.
    """
    now = _now(now)
    today = now.astimezone(EAT).date()
    try:
        fx = fixtures.load_any(
            None, os.path.join(data_root, "fixtures.csv"),
            overlay_dir=os.path.join(here, "data", "manual", "fixtures"))
    except Exception as e:
        return {"key": "schedule", "label": "Fixtures on the board",
                "level": "stale", "today": 0, "tomorrow": 0, "until": None,
                "note": "could not read the schedule: %r" % e,
                "fix": "python predict.py update", "last": "unreadable"}
    days = pd.to_datetime(fx["Date"], errors="coerce").dt.date
    n_today = int((days == today).sum())
    n_tom = int((days == today + timedelta(days=1)).sum())
    until = max([d for d in days.dropna()], default=None)
    # The board running out is the fault - not a quiet day. No league played
    # for a fortnight in September (an international break) while every source
    # was arriving on time, and that must not read as a broken pipeline.
    if until is None or until < today:
        level = "stale"
        note = ("The schedule ends %s. Nothing is listed from today on."
                % (until.strftime("%d %b") if until else "nowhere"))
    elif until == today:
        level = "warn"
        note = "The schedule ends today - nothing is listed beyond it."
    else:
        level, note = "ok", ""
    return {"key": "schedule", "label": "Fixtures on the board",
            "level": level, "today": n_today, "tomorrow": n_tom,
            "until": until.isoformat() if until else None,
            "last": "%d today, %d tomorrow" % (n_today, n_tom),
            "note": note,
            "fix": "python predict.py update && python predict.py "
                   "refresh-fixtures-api --days 2 --with-yesterday"}


def report(data_root: str, repo: str, here: str, p=None, now=None) -> dict:
    """Every source, how long since it arrived, and what to run if it stopped."""
    now = _now(now)
    src = [schedule_cover(data_root, here, now)]

    src.append(_source(
        "feed", "Results and fixtures (football-data)",
        _newest(glob.glob(os.path.join(data_root, "*.csv"))
                + glob.glob(os.path.join(data_root, "*", "*.csv"))),
        now, fix="python predict.py update"))
    src.append(_source(
        "api_fixtures", "Fixtures and cups (API-Football)",
        os.path.join(here, "data", "manual", fixtures.API_FILE), now,
        warn=API_AGE_WARN, stale=API_AGE_STALE,
        fix="python predict.py refresh-fixtures-api --days 2 --with-yesterday"))
    src.append(_source(
        "tanzania", "NBC Premier League (ligikuu.co.tz)",
        os.path.join(here, "data", "manual", "TZ1.csv"), now,
        fix="python predict.py refresh-tanzania"))
    src.append(_source(
        "api_results", "Final scores (API-Football)",
        os.path.join(here, "data", "manual", fixtures.API_RESULTS_FILE), now,
        warn=API_AGE_WARN, stale=API_AGE_STALE,
        fix="python predict.py refresh-fixtures-api --days 2 --with-yesterday"))
    src.append(_source(
        "record", "Public record", os.path.join(repo, "data", "record",
                                                "predictions.csv"), now,
        fix="python predict.py record-publish"))

    tips = _tips_state(repo, now)
    if tips:
        src.append(tips)
    if p is not None:
        src.append(_results_note(p))

    for s in src:
        s["scope"] = SCOPES.get(s["key"], "data")
    return {"generated": now.isoformat(),
            "overall": _worst([s["level"] for s in src]),
            "data_level": _worst([s["level"] for s in src
                                  if s["scope"] == "data"]),
            "publishing_level": _worst([s["level"] for s in src
                                        if s["scope"] == "publishing"]),
            "sources": src,
            "problems": [s["key"] for s in src if s["level"] != "ok"]}


def _tips_state(repo: str, now: datetime) -> dict | None:
    """Today's tip lists frozen yet? Only a fault once the freeze hour passed."""
    try:
        from . import tipslog
    except Exception:
        return None
    day = now.astimezone(EAT)
    try:
        days = tipslog.frozen_days(tipslog.load(repo))
    except Exception as e:
        return {"key": "tips", "label": "Frozen tip lists", "level": "warn",
                "last": "unreadable", "note": "%r" % e, "age_hours": None,
                "fix": "python predict.py freeze-tips"}
    today = day.strftime("%Y-%m-%d")
    yesterday = (day - timedelta(days=1)).strftime("%Y-%m-%d")
    if today in days:
        level, note = "ok", ""
    elif day.hour < tipslog.FREEZE_HOUR:
        level, note = "ok", "due at %02d:00 EAT" % tipslog.FREEZE_HOUR
    elif yesterday in days:
        level, note = "warn", "today's lists are not frozen yet"
    else:
        level, note = "stale", "no lists frozen today or yesterday"
    return {"key": "tips", "label": "Frozen tip lists", "level": level,
            "age_hours": None, "note": note,
            "last": max(days) if days else "never",
            "fix": "python predict.py freeze-tips"}


def _results_note(p) -> dict:
    """The latest result actually loaded. Information, not a verdict: a league
    in an international break has no newer match to find."""
    try:
        last = pd.to_datetime(p.df["Date"]).max()
        n = int(len(p.df))
    except Exception:
        return {"key": "results", "label": "Results loaded", "level": "warn",
                "last": "unreadable", "age_hours": None, "note": "", "fix": ""}
    return {"key": "results", "label": "Results loaded", "level": "ok",
            "age_hours": None, "last": "to %s" % last.strftime("%d %b %Y"),
            "note": "%d matches across %d divisions"
                    % (n, p.df["Div"].nunique()), "fix": ""}


def headline(rep: dict) -> str:
    """One line for a log, an alert, or the top of a screen."""
    bad = [s for s in rep["sources"] if s["level"] != "ok"]
    if not bad:
        sch = next((s for s in rep["sources"] if s["key"] == "schedule"), None)
        feed = next((s for s in rep["sources"] if s["key"] == "feed"), None)
        return "all sources current%s%s" % (
            (" - %d fixtures tomorrow" % sch["tomorrow"]) if sch else "",
            (", data refreshed %s" % feed["last"]) if feed else "")
    return "; ".join("%s %s (%s)" % (s["label"], s["level"], s["last"])
                     for s in bad)
