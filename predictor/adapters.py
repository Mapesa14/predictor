"""Turn other people's data formats into the one shape the engine reads.

The engine wants a football-data.co.uk style table: Div, Date, HomeTeam,
AwayTeam, FTHG, FTAG and, where available, half-time goals. Anything that can
be coerced into that shape can be a league here, which is what lets leagues
with no commercial feed sit beside the ones that have one.

The first adapter reads openfootball's football.txt league files (public
domain). They carry results and half-time scores, but no shots and no prices,
so those leagues run on the goals model alone - which the engine already
falls back to on its own.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

_DATE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})"
    r"(?:\s+(\d{4}))?\s*$")

# Two layouts appear in the wild, sometimes in the same repository.
# "  20:30  Grazer AK   v RB Salzburg   2-3 (2-3)"   score last
_MATCH_VS = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(?P<home>\S.*?)\s+v\s+(?P<away>\S.*?)\s+"
    r"(?P<hg>\d+)\s*-\s*(?P<ag>\d+)"
    r"(?:\s+\((?P<hh>\d+)\s*-\s*(?P<ah>\d+)\))?\s*$")
# "  20:30  LASK   1-1 (0-1)   Rapid Wien"          score in the middle
_MATCH_MID = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(?P<home>\S.*?)\s+(?P<hg>\d+)\s*-\s*(?P<ag>\d+)"
    r"(?:\s+\((?P<hh>\d+)\s*-\s*(?P<ah>\d+)\))?"
    r"\s+(?P<away>\S.*?)\s*$")

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"], 1)}

_SEASON_SPLIT = re.compile(r"(\d{4})-(\d{2})")
_SEASON_ONE = re.compile(r"(\d{4})")


def _season_years(path: str):
    """(first year, second year) from a filename, for either season style."""
    name = os.path.basename(path)
    m = _SEASON_SPLIT.search(name)
    if m:
        y1 = int(m.group(1))
        return y1, int(str(y1)[:2] + m.group(2))
    m = _SEASON_ONE.search(name)
    if m:                       # a calendar-year season, as in Scandinavia
        y = int(m.group(1))
        return y, y
    return None, None


def parse_football_txt(path: str, div: str) -> pd.DataFrame:
    """Read one openfootball league file into the engine's match schema."""
    y1, y2 = _season_years(path)
    rows, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith(("=", "#", "-")):
                continue
            d = _DATE.match(line)
            if d:
                mon, day, yr = d.group(1), int(d.group(2)), d.group(3)
                month = _MONTHS[mon]
                year = int(yr) if yr else (y1 if month >= 7 else y2)
                cur = (pd.Timestamp(year=year, month=month, day=day)
                       if year else None)
                continue
            if cur is None:
                continue
            body = line.rstrip()
            # Goalscorer lines sit under a result, wrapped in parentheses.
            if body.lstrip().startswith("("):
                continue
            # Trailing notes such as "[awarded]" or "[abandoned]" would
            # otherwise be read as the away team by the score-in-middle form.
            body = re.sub(r"\s*\[[^\]]*\]\s*$", "", body).rstrip()
            m = _MATCH_VS.match(body) or _MATCH_MID.match(body)
            if not m:
                continue
            home, away = m.group("home").strip(), m.group("away").strip()
            # A stray goalscorer fragment can look like a result, and a
            # mis-parse swallows the separator into one of the names.
            if (not home or not away or "'" in home or "'" in away
                    or " v " in home or " v " in away):
                continue
            rows.append({
                "Div": div, "Date": cur, "HomeTeam": home, "AwayTeam": away,
                "FTHG": int(m.group("hg")), "FTAG": int(m.group("ag")),
                "HTHG": int(m.group("hh")) if m.group("hh") else np.nan,
                "HTAG": int(m.group("ah")) if m.group("ah") else np.nan,
            })
    return pd.DataFrame(rows)


def build_csvs(raw_dir: str, out_dir: str) -> dict:
    """Convert every `<DIV>_<season>.txt` under raw_dir into one CSV per league."""
    os.makedirs(out_dir, exist_ok=True)
    by_div: dict[str, list] = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.txt"))):
        div = os.path.basename(path).split("_")[0]
        df = parse_football_txt(path, div)
        if len(df):
            by_div.setdefault(div, []).append(df)
    written, all_merges = {}, {}
    for div, parts in by_div.items():
        df = pd.concat(parts, ignore_index=True)
        df, merges = canonicalise(df)
        all_merges.update(merges)
        df = (df.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"])
                .sort_values("Date"))
        df["FTR"] = np.where(df.FTHG > df.FTAG, "H",
                             np.where(df.FTHG < df.FTAG, "A", "D"))
        df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")
        out = os.path.join(out_dir, "%s.csv" % div)
        df.to_csv(out, index=False)
        written[div] = len(df)
    return written, all_merges


# ----------------------------------------------------- club name canonicals
_ACCENTS = str.maketrans({
    "ä": "a", "á": "a", "à": "a", "â": "a", "å": "a", "ã": "a",
    "ö": "o", "ó": "o", "ø": "o", "ô": "o", "õ": "o",
    "ü": "u", "ú": "u", "ù": "u", "û": "u",
    "é": "e", "è": "e", "ê": "e", "ě": "e",
    "í": "i", "ï": "i", "î": "i",
    "ç": "c", "č": "c", "ć": "c", "ñ": "n", "ň": "n",
    "š": "s", "ś": "s", "ż": "z", "ź": "z", "ž": "z", "ł": "l",
    "ř": "r", "ť": "t", "ď": "d", "ů": "u", "ý": "y", "ß": "s",
    "Ö": "O", "Ä": "A", "Ü": "U", "Š": "S", "Č": "C", "Ž": "Z",
})

# Club-type tokens that some sources include and others leave off.
_TYPE = re.compile(
    r"\b(FC|SC|SK|SV|SG|AC|CF|BK|IF|IL|FK|IK|AIK|GAK|SCR|TSV|WSG|RB|RZ|MSK|"
    r"FCSB|NK|HNK|SD|CD|AS|US|OB|BSC|VfB|VfL|LFC|LKS|GKS)\b", re.I)
# Cities appended by some sources but not others.
_SUFFIX_CITY = re.compile(
    r"\s+(Linz|Wien|Graz|Salzburg|Oslo|Bergen|Praha|Prague|Brno|Plzen|"
    r"Ostrava|Kyiv|Kiev|Lviv|Donetsk|Poltava|Odesa|Odessa)$", re.I)


def _key(name: str) -> str:
    """A comparison key that ignores club-type tokens, accents and hyphens."""
    s = name.translate(_ACCENTS)
    s = s.replace("-", " ").replace(".", " ")
    s = re.sub(r"^\s*\d+\s+", "", s)
    s = _TYPE.sub(" ", s)
    s = " ".join(s.split())
    s = _SUFFIX_CITY.sub("", s).strip()
    return s.casefold()


def canonicalise(df: pd.DataFrame):
    """Collapse spellings of the same club within a division.

    Sources disagree about whether to write LASK or LASK Linz, RB Salzburg or
    FC Salzburg. Left alone, one club becomes two half-rated ones. Returns the
    frame plus the merges made, so they can be eyeballed rather than trusted.
    """
    out = df.copy()
    merges = {}
    for div, part in out.groupby("Div"):
        names = pd.concat([part.HomeTeam, part.AwayTeam])
        counts = names.value_counts()
        groups: dict[str, list] = {}
        for n in counts.index:
            groups.setdefault(_key(n), []).append(n)
        mapping = {}
        for k, variants in groups.items():
            best = max(variants, key=lambda v: counts[v])
            for v in variants:
                mapping[v] = best
            if len(variants) > 1:
                merges["%s: %s" % (div, best)] = sorted(variants)
        m = out.Div == div
        out.loc[m, "HomeTeam"] = out.loc[m, "HomeTeam"].map(
            lambda x: mapping.get(x, x))
        out.loc[m, "AwayTeam"] = out.loc[m, "AwayTeam"].map(
            lambda x: mapping.get(x, x))
    return out, merges
