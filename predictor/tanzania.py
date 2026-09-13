"""The Tanzanian top flight, straight from the league's own site.

Every other competition in the product arrives through football-data.co.uk or
openfootball. Neither carries the NBC Premier League's current season: the
openfootball file stops in June 2026 and football-data has never covered
Africa. The aggregators are worse than useless here - they kept serving the
*previous* Tanzanian season long after this one kicked off, which is how the
league ended up looking dormant in a product built for a user in Dar es Salaam.

ligikuu.co.tz is the league's own site and runs SportsPress, whose event
markup is regular enough to read directly:

    <span class="team-logo logo-odd"  title="Pamba Jiji">   ... home
    <span class="team-logo logo-even" title="TRA United">   ... away
    <time class="sp-event-date" datetime="2026-09-13 16:00:00">
    <h5 class="sp-event-results"> <span class="sp-result">2</span>
                                  <span class="sp-result">0</span>
    <div class="sp-event-league">NBC PREMIER LEAGUE 2026/2027</div>

A played event carries two integer results; an unplayed one carries a single
kick-off time ("4:00 pm"). That one distinction splits the page into results
and fixtures, which go to the two overlays that survive a rebuild:

    data/manual/TZ1.csv           results, merged by adapters.build_csvs
    data/manual/fixtures/TZ1.csv  fixtures, merged by fixtures.load_any

Times are East Africa Time (UTC+3), as published.
"""
from __future__ import annotations

import os
import re
import urllib.request
from datetime import datetime

import pandas as pd

URL = "https://ligikuu.co.tz/"
DIV = "TZ1"
LEAGUE = "NBC PREMIER LEAGUE"          # matched case-insensitively, season-agnostic
TIMEOUT = 60

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# The site spells clubs the short way and inconsistently in case ("pamba Jiji",
# "Coastal union"). These are the names the league's own pages use, mapped onto
# the names the rest of the data already carries. Matching is done on a folded
# key, so only genuinely different names need an entry here.
ALIASES = {
    "azam": "Azam FC",
    "coastal union": "Coastal Union FC",
    "dodoma jiji": "Dodoma Jiji FC",
    "fountain gate": "Singida Fountain Gate FC",
    "geita gold": "Geita Gold FC",
    "jkt tanzania": "JKT Tanzania FC",
    "kagera sugar": "Kagera Sugar FC",
    "kmc": "KMC FC",
    "mashujaa": "Mashujaa FC",
    "mbeya city": "Mbeya City FC",
    "namungo": "Namungo FC",
    "pamba jiji": "Pamba Jiji FC",
    "polisi tanzania": "Polisi Tanzania",
    "simba": "Simba SC",
    "singida bs": "Singida Black Stars FC",
    "singida black stars": "Singida Black Stars FC",
    "tra united": "TRA United SC",
    "tanzania prisons": "Tanzania Prisons FC",
    "mtibwa sugar": "Mtibwa Sugar FC",
    "ihefu": "Ihefu FC",
    "kengold": "KenGold FC",
    "tabora united": "Tabora United FC",
    "young africans": "Young Africans SC",
}

_SUFFIX = re.compile(r"\b(fc|sc)\b", re.I)


def _key(name: str) -> str:
    """Fold case, drop an FC/SC suffix and squash spaces."""
    s = _SUFFIX.sub(" ", str(name)).lower()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def resolve(name: str):
    """The canonical club name, or None if this site name is not recognised.

    Returning None rather than a guess is deliberate. A silent near-match is
    how "Le Mans" once became "Lens"; an unknown club here should stop the
    import and be looked at, not quietly become a different team's rating.
    """
    return ALIASES.get(_key(name))


# --------------------------------------------------------------- fetching
def fetch(url: str = URL, timeout: int = TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- parsing
_TD = re.compile(r"<td>(.*?)</td>", re.S)
_ID = re.compile(r"/event/(\d+)/")
_DATE = re.compile(r'<time class="sp-event-date"[^>]*datetime="([^"]+)"')
_LEAGUE = re.compile(r'<div class="sp-event-league">(.*?)</div>', re.S)
_TITLE = re.compile(r'<h4 class="sp-event-title"[^>]*>\s*<a[^>]*>(.*?)</a>', re.S)
_RESULT = re.compile(r'<span class="sp-result[^"]*">(.*?)</span>', re.S)


def parse(html: str, league: str = LEAGUE) -> pd.DataFrame:
    """Every event of one competition on the page, played or not.

    Columns: Div, Date (Timestamp), Time ("HH:MM" EAT), HomeTeam, AwayTeam,
    FTHG, FTAG (None when unplayed), EventId, and raw_home / raw_away as the
    site spelled them.
    """
    want = league.lower()
    rows, seen = [], set()
    for block in _TD.findall(html):
        if "sp-event-date" not in block:
            continue
        lg = _LEAGUE.search(block)
        if not lg or want not in lg.group(1).lower():
            continue
        eid = _ID.search(block)
        eid = eid.group(1) if eid else None
        if eid in seen:            # the same event appears in several widgets
            continue
        seen.add(eid)

        when = _DATE.search(block)
        title = _TITLE.search(block)
        if not when or not title:
            continue
        text = re.sub(r"\s+", " ", title.group(1)).strip()
        low = text.lower()
        if " vs " not in low:
            continue
        i = low.index(" vs ")
        raw_h, raw_a = text[:i].strip(), text[i + 4:].strip()

        ts = pd.to_datetime(when.group(1), errors="coerce")
        if pd.isna(ts):
            continue

        parts = [p.strip() for p in _RESULT.findall(block)]
        goals = [int(p) for p in parts if re.fullmatch(r"\d+", p)]
        fthg, ftag = (goals[0], goals[1]) if len(goals) >= 2 else (None, None)

        rows.append({
            "Div": DIV,
            "Date": ts.normalize(),
            "Time": ts.strftime("%H:%M"),
            "HomeTeam": resolve(raw_h),
            "AwayTeam": resolve(raw_a),
            "FTHG": fthg,
            "FTAG": ftag,
            "EventId": eid,
            "raw_home": raw_h,
            "raw_away": raw_a,
        })
    df = pd.DataFrame(rows, columns=["Div", "Date", "Time", "HomeTeam", "AwayTeam",
                                     "FTHG", "FTAG", "EventId", "raw_home", "raw_away"])
    return df.sort_values("Date").reset_index(drop=True)


def unknown_clubs(df: pd.DataFrame) -> list:
    """Site spellings that `resolve` could not place, for a loud report."""
    bad = set()
    for side, raw in (("HomeTeam", "raw_home"), ("AwayTeam", "raw_away")):
        miss = df[df[side].isna()]
        bad.update(str(v) for v in miss[raw])
    return sorted(bad)


# --------------------------------------------------------------- writing
def _results_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["FTHG"].notna() & df["FTAG"].notna()].copy()
    out = d[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)
    out["HTHG"] = ""          # the site publishes no half-time score
    out["HTAG"] = ""
    out["Date"] = out["Date"].dt.strftime("%d/%m/%Y")
    return out.reset_index(drop=True)


def _fixtures_frame(df: pd.DataFrame, as_of=None, played=None) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of or datetime.now()).normalize()
    d = df[df["FTHG"].isna() & (df["Date"] >= as_of)].copy()
    if played is not None and len(played):
        # A pairing we already hold a result for is not a fixture, however the
        # site lists it. Belt and braces: the homepage does not currently
        # repeat a played event, but a stale widget once would be enough to
        # put a finished match on the user's card as if it were tonight's.
        done = set(zip(played["HomeTeam"], played["AwayTeam"]))
        # a Series, not a bare list: `frame[[]]` selects no *columns*
        keep = pd.Series([(h, a) not in done
                          for h, a in zip(d["HomeTeam"], d["AwayTeam"])],
                         index=d.index, dtype=bool)
        d = d[keep]
    out = d[["Div", "Date", "Time", "HomeTeam", "AwayTeam"]].copy()
    out["Date"] = out["Date"].dt.strftime("%d/%m/%Y")
    return out.reset_index(drop=True)


def _merge_results(path: str, fresh: pd.DataFrame) -> pd.DataFrame:
    """Add to what the overlay already holds; never shrink it.

    The homepage publishes a window, not the whole season - it carried eight
    of the ten results already on file. Replacing the overlay with a scrape
    would have quietly deleted two verified matches, so the scrape merges in
    and wins only where the same fixture appears on both sides.
    """
    if not os.path.isfile(path):
        return fresh
    old = pd.read_csv(path, dtype=str).fillna("")
    if not len(old):
        return fresh
    both = pd.concat([old, fresh.astype(str)], ignore_index=True)
    both = both.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"],
                                keep="last")
    both["_d"] = pd.to_datetime(both["Date"], format="%d/%m/%Y", errors="coerce")
    return both.sort_values("_d").drop(columns="_d").reset_index(drop=True)


def sync(root: str, html: str = None, url: str = URL, as_of=None) -> dict:
    """Read the league site and refresh both Tanzanian overlays.

    Returns a summary; raises ValueError if a club on the page cannot be
    resolved, so an unknown name is never dropped or guessed at silently.
    """
    html = html if html is not None else fetch(url)
    df = parse(html)
    if df.empty:
        raise ValueError("no %s events found at %s" % (LEAGUE, url))
    bad = unknown_clubs(df)
    if bad:
        raise ValueError("unrecognised club(s) on %s: %s - add them to "
                         "tanzania.ALIASES" % (url, ", ".join(bad)))

    res_path = os.path.join(root, "data", "manual", "%s.csv" % DIV)
    scraped = _results_frame(df)
    res = _merge_results(res_path, scraped)
    fix = _fixtures_frame(df, as_of, played=res)

    fix_path = os.path.join(root, "data", "manual", "fixtures", "%s.csv" % DIV)
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    os.makedirs(os.path.dirname(fix_path), exist_ok=True)
    res.to_csv(res_path, index=False)
    fix.to_csv(fix_path, index=False)
    return {"events": int(len(df)), "results": int(len(res)),
            "scraped_results": int(len(scraped)),
            "fixtures": int(len(fix)), "results_file": res_path,
            "fixtures_file": fix_path,
            "latest_result": res["Date"].iloc[-1] if len(res) else "",
            "next_fixture": fix["Date"].iloc[0] if len(fix) else ""}
