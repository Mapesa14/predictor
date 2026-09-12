"""Where the next fixtures come from.

Three sources, in order of preference:
  1. a fixtures CSV you point at (`--fixtures path.csv`),
  2. football-data.co.uk's weekly fixtures feed (`refresh`, network required),
  3. the round-robin remainder worked out from results already in the data.

Source 3 needs no network at all: in a league where everyone plays everyone
home and away, whatever pairing has not been played yet is still to come.
"""
from __future__ import annotations

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
# replaces the date and time when the schedule is not yet fixed.
EXTRA_COLS = ["Leg1H", "Leg1A", "WhenNote"]


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


def load_any(path: str | None, cache: str) -> pd.DataFrame:
    """Read the given fixtures file, else a previously downloaded cache, else empty."""
    for p in (path, cache):
        if p and os.path.exists(p):
            try:
                return from_csv(p)
            except Exception:
                continue
    return pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam"])


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
             days: int = 14, as_of: datetime | None = None) -> pd.DataFrame:
    """Scheduled fixtures inside the next `days`, falling back to the round robin."""
    as_of = as_of or datetime.now()
    if len(fx):
        f = fx[(fx["Date"] >= pd.Timestamp(as_of).normalize()) &
               (fx["Date"] <= pd.Timestamp(as_of) + pd.Timedelta(days=days))]
        if div:
            f = f[f["Div"] == div]
        if len(f):
            return f.reset_index(drop=True)
    divs = [div] if div else sorted(set(df["Div"]) & set(leagues.LEAGUES))
    parts = [remaining(df, d) for d in divs]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["Div", "Date", "HomeTeam", "AwayTeam", "note"])
