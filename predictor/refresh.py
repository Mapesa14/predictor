"""Pull the latest European results in; nothing is stashed, nothing lost.

football-data.co.uk publishes one CSV per league per season and re-fills the
current season's rows as matches are played, so re-downloading that one file
and merging it with the history already sorted on disk keeps the models
current. Merging is additive: older seasons stay, fresh results win, and the
result is written back into the same place.
"""
from __future__ import annotations

import glob
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd

from . import loader

MMZ = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
TOP = 60  # seconds, same as fixtures refresh


def current_season(now: datetime | None = None) -> str:
    now = now or datetime.now()
    start = now.year if now.month >= 7 else now.year - 1
    return "%02d%02d" % (start % 100, (start + 1) % 100)


def download(div: str, season: str) -> pd.DataFrame | None:
    """The current season's file as already-normalised results, or None."""
    url = MMZ.format(season=season, div=div)
    try:
        with urllib.request.urlopen(url, timeout=TOP) as r:
            raw = r.read()
    except urllib.error.HTTPError:
        return None
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        df = loader._read_one(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if df is None or not len(df):
        return None
    # results only: the rest of the season's fixture rows are blank until played
    return df[df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
              .notna().all(axis=1)]


def refresh_footballdata(root: str, divs: list[str], season: str | None = None,
                         since: int | None = None) -> list[dict]:
    """Merge current (and optionally earlier) seasons into each local CSV.

    `since` is a start year, e.g. 2023 pulls every season from 2023-24 to the
    current one, which re-establishes the multi-season history a single-file
    division keeps in one CSV.
    """
    season = season or current_season()
    years = [season]
    if since is not None:
        start = 2000 + int(season[:2])
        years = ["%02d%02d" % (y % 100, (y + 1) % 100)
                 for y in range(max(int(since), start - 6), start + 1)]
    keep = loader.CORE + loader.EXTRA + loader.ODDS
    out = []
    for div in divs:
        parts = []
        for yrs in years:
            f = download(div, yrs)
            if f is not None and len(f):
                parts.append(f)
        if not parts:
            out.append({"div": div, "file": "", "added": 0, "total": 0,
                        "note": "football-data has nothing for %s yet" % div})
            continue
        fresh = pd.concat(parts, ignore_index=True)
        existing = loader.load(root)
        existing = existing[existing["Div"] == div]
        merged = (pd.concat([existing, fresh], ignore_index=True)
                    .drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"],
                                     keep="last")
                    .sort_values(["Div", "Date"])
                    .reset_index(drop=True))
        merged = merged[keep] if len(merged) else merged

        # A refresh may only add: if merging would shrink what the engine sees
        # on disk, something is wrong with the pull - refuse to clobber.
        if len(existing) and len(merged) < len(existing) * 0.9:
            out.append({"div": div, "file": "", "added": 0,
                        "total": int(len(existing)),
                        "note": "refusing: merge (%d) shrank existing (%d)"
                        % (len(merged), len(existing))})
            continue

        # Write where this division already lives: Eng/ seasons are one folder
        # per season, the rest are one combined file in a country folder.
        home = glob.glob(os.path.join(root, "**", "%s.csv" % div),
                         recursive=True)
        if home:
            folder = os.path.dirname(home[0])
            if len(os.path.basename(folder)) == 4 and \
                    os.path.basename(folder).isdigit():
                dest_dir = os.path.join(os.path.dirname(folder), season)
            else:
                dest_dir = folder
        else:
            dest_dir = root
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "%s.csv" % div)
        # The loader parses %d/%m/%Y; a raw Timestamp column would be written
        # as ISO-8601 and read back as all-NaT (and so silently dropped).
        to_write = merged.copy()
        to_write["Date"] = to_write["Date"].dt.strftime("%d/%m/%Y")
        to_write.to_csv(dest, index=False)
        out.append({"div": div, "file": dest, "added": int(len(fresh)),
                    "total": int(len(merged)),
                    "note": "" if len(fresh) else "no new rows"})
    return out


DIVS = ["E0", "E1", "E2", "E3", "EC", "D1", "D2", "SP1", "SP2",
        "I1", "I2", "F1", "F2", "N1", "N2", "B1", "B2",
        "T1", "G1", "SC0", "SC1", "SC2", "SC3", "P1", "P2"]


# --------------------------------------------------- second source: openfootball
# African federations have no football-data feed, but the public-domain
# openfootball/world repo carries them, one txt file per season such as
#   africa/tanzania/2025-26_tz1.txt
# The engine's football.txt adapter (adapters.py) turns those into the same
# Div, Date, HomeTeam, AwayTeam, FTHG, FTAG league shape as everything else.
OPENFOOTBALL_WORLD = {
    "TZ1": "tanzania", "EG1": "egypt", "DZ1": "algeria", "MA1": "morocco",
    "ZA1": "south-africa", "NG1": "nigeria", "GH1": "ghana", "KE1": "kenya",
    "UG1": "uganda", "ZM1": "zambia", "RW1": "rwanda",
}


def _github_dir(repo: str, path: str) -> list[dict]:
    import urllib.request as _ur
    url = "https://api.github.com/repos/%s/contents/%s" % (repo, path)
    req = _ur.Request(url, headers={"User-Agent": "football-predictor"})
    with _ur.urlopen(req, timeout=TOP) as r:
        return json.loads(r.read().decode("utf-8"))


def _latest_openfootball(raw_dir: str, out_dir: str, div: str) -> dict:
    """Grab this division's newest openfootball season into data/raw, then
    rebuild every CSV that build_csvs owns (additive: earlier seasons from the
    same repo folder stay). Returns what changed.
    """
    from . import adapters
    folder = OPENFOOTBALL_WORLD.get(div)
    if not folder:
        raise ValueError("openfootball source covers only: %s"
                         % ", ".join(sorted(OPENFOOTBALL_WORLD)))
    base = "africa/%s" % folder
    entries = [e for e in _github_dir("openfootball/world", base)
               if e["type"] == "file" and e["name"].endswith(".txt")]
    if not entries:
        raise ValueError("no league files under %s" % base)
    name = sorted(entries, key=lambda e: e["name"], reverse=True)[0]["name"]

    m = re.search(r"(\d{4})-(\d{2})", name)          # 2025-26_tz1.txt
    season = "%d-%02d" % (int(m.group(1)), int(m.group(2))) \
        if m else "latest"
    dest = os.path.join(raw_dir, "%s_%s.txt" % (div, season))
    if not os.path.exists(dest):
        url = ("https://raw.githubusercontent.com/openfootball/"
               "world/master/%s/%s" % (base, name))
        with urllib.request.urlopen(url, timeout=TOP) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
    written, merges = adapters.build_csvs(raw_dir, out_dir)
    return {"source": "openfootball/%s/%s" % (base, name),
            "raw": dest, "written": written, "merges": merges}


def refresh_all(root: str, since: int | None = None) -> list[dict]:
    """football-data results for every division we track, then the fixtures
    feed - the two things a running product needs to stay self-learning."""
    from . import fixtures
    out = refresh_footballdata(root, DIVS, since=since)
    dest = os.path.join(root, "fixtures.csv")
    fixtures.refresh(dest)
    fx = fixtures.from_csv(dest)
    out.append({"div": "FIX", "file": dest, "added": int(len(fx)),
                "total": int(len(fx)),
                "note": "upcoming fixtures feed"})
    return out