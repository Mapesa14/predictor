"""Load and normalise football-data.co.uk result CSVs."""
from __future__ import annotations

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

from . import leagues

CORE = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG"]
EXTRA = ["Time", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
# Consensus closing odds, used only for value comparison / backtesting.
ODDS = ["AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5", "B365H", "B365D", "B365A"]


def _parse_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = d.isna()
    if missing.any():  # some seasons use 2-digit years
        d[missing] = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
    return d


def _read_one(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)
        except Exception:
            return None
    df.columns = [c.strip() for c in df.columns]
    if not set(CORE[:7]).issubset(df.columns):
        return None
    keep = [c for c in CORE + EXTRA + ODDS if c in df.columns]
    df = df[keep].copy()
    df["Date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    for c in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["Div"] = df["Div"].astype(str).str.strip()
    df["source"] = os.path.basename(path)
    return df


def season_of(d: datetime) -> str:
    """Football season label: Aug-May, so Jan-Jun belongs to the previous start year."""
    y = d.year
    start = y if d.month >= 7 else y - 1
    return f"{start}/{str(start + 1)[-2:]}"


# Leagues converted from other sources live beside the package, so they travel
# with the code rather than depending on the user's download folder.
BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "leagues")


def load(root: str, divs: list[str] | None = None,
         extra_roots: list[str] | None = None) -> pd.DataFrame:
    """Read every CSV under `root` (and any extra roots), return one table."""
    roots = [root] + list(extra_roots if extra_roots is not None
                          else ([BUNDLED] if os.path.isdir(BUNDLED) else []))
    files = []
    for r in roots:
        files += glob.glob(os.path.join(r, "**", "*.csv"), recursive=True)
    files = sorted(set(files))
    frames = [f for f in (_read_one(p) for p in files) if f is not None and len(f)]
    if not frames:
        raise SystemExit(f"No usable football-data CSVs found under {root}")
    df = pd.concat(frames, ignore_index=True)

    # The same fixture can appear in several downloads; keep the richest copy.
    df["_fill"] = df.notna().sum(axis=1)
    df = (df.sort_values("_fill", ascending=False)
            .drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"], keep="first")
            .drop(columns="_fill"))

    if divs:
        df = df[df["Div"].isin(divs)]
    df = df[df["Div"].isin(leagues.LEAGUES)]
    df["Season"] = df["Date"].map(season_of)
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H",
                np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    # Point-in-time schema (§5.5): a result is only knowable from its match
    # date onwards. Every consumer should read the table through as_of() so
    # nothing is ever trained on a fact that did not exist yet.
    df["known_at"] = df["Date"]
    return df.sort_values(["Div", "Date"]).reset_index(drop=True)


def as_of(df: pd.DataFrame, moment) -> pd.DataFrame:
    """Rows whose facts were known by `moment` - the point-in-time filter.

    Answer "what did we know at kick-off minus 60 minutes" by calling this with
    `moment = kickoff - 60min`. A missing `known_at` is a schema error, not a
    silent fallback: leaking the future into a backtest is the failure mode
    this column exists to make impossible.
    """
    if "known_at" not in df.columns:
        raise ValueError("table carries no `known_at`; run it through load()")
    return df[df["known_at"] <= pd.Timestamp(moment)].copy()


def teams(df: pd.DataFrame, div: str | None = None) -> list[str]:
    d = df[df["Div"] == div] if div else df
    return sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
