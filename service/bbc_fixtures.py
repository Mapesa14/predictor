"""European cup fixtures from BBC Sport, because no other source we have them.

football-data.co.uk is domestic leagues only: it has never carried a Champions
League tie. API-Football did, and when that account was suspended the entire
European programme vanished from the product - on a Wednesday with Manchester
City v Paris Saint-Germain and Roma v Real Madrid on the board. A product that
cannot show those is not a football product.

BBC Sport publishes the day's fixtures as server-rendered HTML with no key, no
quota and no account to suspend, and every fixture carries one unambiguous
line:

    Manchester City versus Paris Saint-Germain kick off 20:00

That line, and the competition heading above it, are all this parser uses. It
deliberately takes **only** the competitions nothing else covers: the domestic
leagues keep coming from football-data, which carries closing prices that BBC
does not, and two sources for one fixture is how a schedule starts
contradicting itself.

Clubs are matched the way every other source is matched here - an alias checked
by hand, the exact name, or one generic word dropped leaving exactly one club -
and a tie is priced by the cross-league bridge, each club on its own league's
ratings. A name that does not resolve is reported, never guessed.
"""
from __future__ import annotations

import html as _html
import os
import re
import unicodedata
from datetime import timedelta

import pandas as pd

from predictor import fixtures, leagues
from service import fixtures_api, live

BASE = "https://www.bbc.com/sport/football/scores-fixtures/%s"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) football-predictor/1.0 "
      "(+https://github.com/Mapesa14/predictor)")
TIMEOUT = 25

# Only what no other source gives us. The slug in the BBC's own table link is
# the stable key; the visible name is decoration and gets reworded every year.
COMPETITIONS = {
    "champions-league": "Champions League",
    "europa-league": "Europa League",
    "europa-conference-league": "Conference League",
    "conference-league": "Conference League",
}

COLUMNS = fixtures_api.COLUMNS

_HEAD = re.compile(r'<h2[^>]*GroupHeader[^>]*>.*?href="/sport/football/'
                   r'([a-z0-9-]+)/(?:table|scores-fixtures)"', re.S)
_GAME = re.compile(r'>([^<>]{2,60}?) versus ([^<>]{2,60}?) kick off '
                   r'(\d{1,2}:\d{2})<')


def file(root: str) -> str:
    return os.path.join(root, "data", "manual", fixtures.BBC_FILE)


def fetch_day(day: str, transport=None) -> str:
    """One day's fixtures page. No key, no budget - nothing to spend."""
    if transport is not None:
        return transport(BASE % day)
    status, _headers, raw = live.http_get(BASE % day, {"User-Agent": UA}, TIMEOUT)
    if status != 200:
        raise RuntimeError("BBC answered HTTP %s for %s" % (status, day))
    return raw.decode("utf-8", "replace")


def _clean(s: str) -> str:
    s = _html.unescape(s or "").replace(" ", " ")
    return " ".join(s.split())


def parse(page: str) -> list:
    """[{comp, slug, home, away, time}] for the competitions we take."""
    heads = [(m.start(), m.group(1)) for m in _HEAD.finditer(page)]
    out = []
    for m in _GAME.finditer(page):
        slug = next((s for pos, s in reversed(heads) if pos < m.start()), None)
        comp = COMPETITIONS.get(slug)
        if not comp:
            continue
        home, away, t = (_clean(x) for x in m.groups())
        if home and away:
            out.append({"comp": comp, "slug": slug, "home": home,
                        "away": away, "time": t if len(t) == 5 else "0" + t})
    return out


def build_rows(items, p, day: str) -> tuple:
    """Fixtures-file rows, and what could not be matched.

    Kick-offs are published in UK time, which is how every European division's
    file is already read (leagues.source_tz), so they are stored as given.
    """
    rows, report = [], {"unresolved": [], "no_bridge": []}
    when = pd.Timestamp(day).strftime("%d/%m/%Y")
    for it in items:
        h = fixtures_api.resolve_europe(p, it["home"])
        a = fixtures_api.resolve_europe(p, it["away"])
        missing = [n for n, r in ((it["home"], h), (it["away"], a)) if r is None]
        if missing:
            report["unresolved"].append("%s v %s (%s): %s" % (
                it["home"], it["away"], it["comp"], ", ".join(missing)))
            continue
        rows.append({"Div": h[1], "Date": when, "Time": it["time"],
                     "HomeTeam": h[0], "AwayTeam": a[0], "Comp": it["comp"],
                     "ScaleAlt": "", "AwayDiv": a[1]})
    return rows, report


def sync(root: str, p, days: int = 3, now=None, transport=None) -> dict:
    """Fetch the next few days and rewrite the BBC fixtures file.

    A day that could not be fetched keeps whatever the file already had, so a
    bad response never empties the schedule - the same rule as every other
    source here.
    """
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(max(1, int(days)))]
    rows, fetched, errors = [], [], []
    report = {"unresolved": [], "no_bridge": []}
    for day in dates:
        try:
            page = fetch_day(day, transport)
        except Exception as e:
            errors.append("%s: %s" % (day, e))
            continue
        fetched.append(day)
        r, rep = build_rows(parse(page), p, day)
        rows += r
        for k, v in rep.items():
            report[k].extend(v)

    path = file(root)
    out = pd.DataFrame(rows, columns=COLUMNS)
    if fetched:
        if os.path.isfile(path):
            old = pd.read_csv(path, dtype=str, keep_default_na=False,
                              na_values=[""]).fillna("")
            keep_days = {pd.Timestamp(d).strftime("%d/%m/%Y")
                         for d in set(dates) - set(fetched)}
            keep = old[old["Date"].isin(keep_days)] if keep_days else old.iloc[0:0]
            out = pd.concat([keep.reindex(columns=COLUMNS), out], ignore_index=True)
        out = out.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"],
                                  keep="last")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        out.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return {"fetched": fetched, "errors": errors, "rows": int(len(out)),
            "written": bool(fetched), "file": path, "report": report,
            "by_comp": dict(pd.Series([r["Comp"] for r in rows]).value_counts())
            if rows else {}}
