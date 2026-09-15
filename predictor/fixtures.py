"""Where the next fixtures come from.

Three sources, in order of preference:
  1. a fixtures CSV you point at (`--fixtures path.csv`),
  2. football-data.co.uk's weekly fixtures feed (`refresh`, network required),
  3. the round-robin remainder worked out from results already in the data.

Source 3 needs no network at all: in a league where everyone plays everyone
home and away, whatever pairing has not been played yet is still to come.

On top of whichever of 1-2 is used sits `data/manual/fixtures/`, an overlay for
competitions the European feed does not cover at all (see `overlay`).
"""
from __future__ import annotations

import glob
import os
from datetime import datetime

import pandas as pd

from . import leagues, loader

FEED_URL = "https://football-data.co.uk/fixtures.csv"
SPLIT_LEAGUES = {"SC0", "B1", "G1"}   # championship/relegation splits, not a pure round robin


# Price columns are carried through so the market prior can be applied to a
# fixture that has not been played yet.
ODDS_COLS = ["AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A",
             "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA",
             "Avg>2.5", "Avg<2.5", "B365>2.5", "B365<2.5", "Max>2.5", "Max<2.5"]


# Optional columns a hand-built fixtures file may carry. Leg1H / Leg1A are the
# goals the second-leg home and away sides scored in a first leg; WhenNote
# replaces the date and time when the schedule is not yet fixed; Comp names a
# cup competition whose tie is priced on the division given in Div; ScaleAlt
# is the lower division of a tie across divisions, priced a second time so the
# spread between the two scales can be shown.
EXTRA_COLS = ["Leg1H", "Leg1A", "WhenNote", "Comp", "ScaleAlt"]

# Fixtures fetched from API-Football by service/fixtures_api.py, beside the
# hand-kept overlay directory.
API_FILE = "fixtures_api.csv"


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    need = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if not need.issubset(df.columns):
        raise ValueError("fixtures file needs columns: " + ", ".join(sorted(need)))
    keep = ["Div", "Date", "Time", "HomeTeam", "AwayTeam"] + ODDS_COLS + EXTRA_COLS
    out = df[[c for c in keep if c in df.columns]].copy()
    for c in ODDS_COLS + ["Leg1H", "Leg1A"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Time" in out.columns:
        # a blank kick-off time means "not published", never the text "nan"
        out["Time"] = out["Time"].where(out["Time"].notna(), None)
    out["Date"] = loader._parse_dates(out["Date"])
    # A fixture row is knowable from the moment its feed was read, which is
    # what the point-in-time schema (§5.5) records. Consumers that ask "what
    # did we know yesterday" can pass a moment and get honest nothing for a
    # slate downloaded today.
    out["known_at"] = pd.Timestamp(datetime.now())
    for c in ("Div", "HomeTeam", "AwayTeam"):
        out[c] = out[c].astype(str).str.strip()
    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def from_csv(path: str) -> pd.DataFrame:
    return _norm(pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip"))


def refresh(dest: str, url: str = FEED_URL) -> str:
    """Download the upcoming-fixtures feed. Explicitly invoked, never automatic."""
    import urllib.request
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def overlay(overlay_dir: str | None) -> pd.DataFrame:
    """Hand-kept fixtures for competitions the feed does not carry.

    football-data's feed is European. Anything else with a real published
    schedule - the Tanzanian top flight above all - is written here by its own
    adapter and merged on every load, because `refresh` overwrites the cache
    wholesale and would otherwise drop those competitions back to bare
    round-robin pairings with no date, time or price.
    """
    empty = pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam"])
    if not overlay_dir or not os.path.isdir(overlay_dir):
        return empty
    parts = []
    for p in sorted(glob.glob(os.path.join(overlay_dir, "*.csv"))):
        try:
            f = from_csv(p)
        except Exception:
            continue
        if len(f):
            parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else empty


def default_overlay_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "manual", "fixtures")


def load_any(path: str | None, cache: str, overlay_dir: str | None = None,
             api_file: str | None = None) -> pd.DataFrame:
    """The schedule: a fixtures file or the downloaded feed, gaps filled from
    API-Football, and the hand-kept league overlays on top.

    When one fixture appears in more than one source the more authoritative
    copy is kept: a league's own site (the overlay) over football-data, which
    carries closing prices, over API-Football, which only fills what neither
    has. The API file sits beside the overlay directory, so a test that passes
    its own overlay directory never picks up a real one.
    """
    empty = pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam"])
    if overlay_dir is None:
        overlay_dir = default_overlay_dir()
    if api_file is None:
        api_file = os.path.join(os.path.dirname(overlay_dir), API_FILE)
    base = empty
    for p in (path, cache):
        if p and os.path.exists(p):
            try:
                base = from_csv(p)
                break
            except Exception:
                continue
    api = empty
    if api_file and os.path.isfile(api_file):
        try:
            api = from_csv(api_file)
        except Exception:
            api = empty
    extra = overlay(overlay_dir)
    parts = [df for df in (api, base, extra) if len(df)]
    if not parts:
        return base
    if len(parts) == 1:
        return parts[0]
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"],
                                    keep="last")
    return merged.sort_values("Date").reset_index(drop=True)


def current_season(df: pd.DataFrame, div: str) -> str:
    d = df[df["Div"] == div]
    return d["Season"].max() if len(d) else ""


def remaining(df: pd.DataFrame, div: str) -> pd.DataFrame:
    """Fixtures still outstanding in the current season, from the round-robin."""
    season = current_season(df, div)
    d = df[(df["Div"] == div) & (df["Season"] == season)]
    cols = ["Div", "Date", "HomeTeam", "AwayTeam", "note"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    played = set(zip(d["HomeTeam"], d["AwayTeam"]))
    pairs = [(h, a) for h in teams for a in teams
             if h != a and (h, a) not in played]
    rows = [{"Div": div, "HomeTeam": h, "AwayTeam": a, "Date": pd.NaT,
             "note": "unplayed pairing"} for h, a in _matchdays(pairs)]
    out = pd.DataFrame(rows, columns=cols)
    if div in SPLIT_LEAGUES and len(out):
        out["note"] = "unplayed pairing (split-format league, may not be scheduled)"
    return out


def _matchdays(pairs):
    """Order loose pairings into rounds so no team appears twice in a round."""
    left, out = list(pairs), []
    while left:
        used, rest = set(), []
        for h, a in left:
            if h in used or a in used:
                rest.append((h, a))
            else:
                used.update((h, a))
                out.append((h, a))
        left = rest
    return out


def upcoming(df: pd.DataFrame, fx: pd.DataFrame, div: str | None = None,
             days: int = 14, as_of: datetime | None = None,
             remaining_fn=None) -> pd.DataFrame:
    """Scheduled fixtures inside the next `days`, falling back to the round robin.

    `remaining_fn(df, div)` lets a caller substitute a cached round robin. The
    remainder depends only on results already on disk, so recomputing it for
    every division on every request - two thousand pairings, ordered into
    rounds, to be counted and thrown away - was half the cost of a slate.
    """
    as_of = as_of or datetime.now()
    rem = remaining_fn or remaining
    if len(fx):
        f = fx[(fx["Date"] >= pd.Timestamp(as_of).normalize()) &
               (fx["Date"] <= pd.Timestamp(as_of) + pd.Timedelta(days=days))]
        if div:
            f = f[f["Div"] == div]
        if len(f):
            return f.reset_index(drop=True)
    divs = [div] if div else sorted(set(df["Div"]) & set(leagues.LEAGUES))
    parts = [rem(df, d) for d in divs]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["Div", "Date", "HomeTeam", "AwayTeam", "note"])
