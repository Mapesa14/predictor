"""Live scores via API-Football (api-sports.io) v3.

Nothing is persisted: results live only in an in-memory TTL cache, so a restart
or a missed request costs nothing locally. Configure with:

    LIVE_API_KEY=<key>        required; api-sports.io dashboard key
    LIVE_LEAGUES=39,140,...   optional; league ids to poll, defaults below
    LIVE_BASE_URL=...         optional; override the endpoint

League ids are best-guesses to keep the request budget small; the free tier is
100 requests/day and every (league, date) poll is one request. Use
`/api/live/leagues?search=tanzania` to find correct ids and, if needed, set
LIVE_LEAGUES to the right ones.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

BASE = "https://v3.football.api-sports.io"

# api-football league ids per product division. The European top tens are
# confident; deep-African ids are best guesses and are cheap to correct via
# /api/live/leagues?search=.
DIV_LEAGUES = {
    "E0": 39, "E1": 40, "E2": 41, "E3": 42, "EC": 43,
    "D1": 78, "D2": 79,
    "I1": 135, "I2": 136,
    "F1": 61, "F2": 62,
    "SP1": 140, "SP2": 141,
    "N1": 88, "N2": 89,
    "B1": 144, "B2": 584,
    "T1": 203,
    "G1": 197,
    "SC0": 179, "SC1": 180, "SC2": 181, "SC3": 182,
    "P1": 94, "P2": 297,
    "TZ1": 509,
    "EG1": 233, "DZ1": 340, "MA1": 200, "ZA1": 368,
    "NG1": 157, "GH1": 265, "KE1": 1085, "UG1": 424,
    "ZM1": 948, "RW1": 244,
    "CAFCC": 691, "CAFCL": 690,
}
DEFAULT_LEAGUES = sorted(set(DIV_LEAGUES.values()))
TTL = 60                                    # seconds between provider polls
POLL = {"date"}                             # cache key: one entry per date

STATUS_LIVE = {"1H", "HT", "2H", "ET", "BT", "P", "AET", "PEN", "INT"}  # noqa: E501
STATUS_FT = {"FT", "AET", "PEN"}            # finished (incl. decided in ET/pen)


def configured(key=None) -> bool:
    return bool(key or os.environ.get("LIVE_API_KEY", "").strip())


def _headers(key):
    return {
        "x-apisports-key": key or os.environ["LIVE_API_KEY"],
        "Accept": "application/json",
    }


def _get(path, params, key, timeout=20):
    q = urllib.parse.urlencode(params)
    url = BASE + path + ("?" + q if q else "")
    req = urllib.request.Request(url, headers=_headers(key))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LiveFetcher:
    def __init__(self, key, leagues=None, ttl=TTL):
        self.key = key
        self.leagues = leagues if leagues else DEFAULT_LEAGUES
        self.ttl = ttl
        self._cache = {}            # (league, date) -> (stamp, matches)
        self._lock = None

    def fetch_date(self, date_str: str):
        """Live + finished rows for `date_str` (YYYY-MM-DD) across poll leagues."""
        out = []
        for lid in self.leagues:
            now = time.time()
            c = self._cache.get((lid, date_str))
            if c and now - c[0] < self.ttl:
                matches = c[1]
            else:
                matches = self._poll(lid, date_str)
                self._cache[(lid, date_str)] = (now, matches)
            out.extend(matches)
        out.sort(key=lambda m: m["date"])
        return out

    def _poll(self, lid, date_str):
        try:
            body = _get("/fixtures", {"date": date_str, "league": lid,
                                      "status": "NS-ST-FT"}, self.key)
        except Exception:
            return []
        out = []
        for fix in body.get("response", []) or []:
            try:
                fx = fix["fixture"]; tm = fix["teams"]; gl = fix["goals"]
                hts, ats = tm["home"]["name"], tm["away"]["name"]
                if not hts or not ats:
                    continue
                out.append({
                    "league": lid,
                    "date": fx.get("date"),
                    "status": fx.get("status").get("short"),
                    "minute": (fx.get("status") or {}).get("elapsed"),
                    "home": hts, "away": ats,
                    "hg": gl.get("home") if gl.get("home") is not None else 0,
                    "ag": gl.get("away") if gl.get("away") is not None else 0,
                })
            except (KeyError, TypeError, IndexError):
                continue
        return out


def find_leagues(query, key, limit=8) -> list[dict]:
    """Search the provider's leagues (e.g. 'tanzania') to fix DIV_LEAGUES."""
    body = _get("/leagues", {"search": query}, key)
    out = []
    for row in (body.get("response") or [])[:limit]:
        l = row["league"]; c = row["country"]
        out.append({"id": l["id"], "name": l["name"], "type": l["type"],
                    "country": c.get("name")})
    return out