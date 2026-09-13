"""Live scores from API-Football, spent like the scarce thing they are.

The free tier is 100 requests a day *and* 10 a minute, and going over does not
simply fail: API-Football may temporarily block the key or the server's IP.
The previous version of this module polled all 38 configured leagues one
request each, every minute, on the user's request path. A single cold page
load was 38 calls - nearly four times the per-minute cap - and two of them
spent the whole day.

So the flow is now:

    Frontend  ->  /api/live  ->  database snapshot
                                        ^
                  one refresher, only while a match we carry is in play
                                        |
                                  API-Football

Four rules keep it inside the budget:

  1. **The frontend never reaches the provider.** /api/live reads the last
     snapshot and nothing else, however many clients poll it.
  2. **No call unless something is in play.** Our own schedule knows every
     kick-off. Outside [kick-off - 5 min, kick-off + 130 min] for every fixture
     we carry, the refresher spends nothing - which is most of every day.
  3. **One request per poll.** `fixtures?live=all` returns every live match in
     every league at once, instead of one request per league.
  4. **Every attempt is charged before it is sent**, to a ledger in the
     database, against a rolling 24-hour window. Rolling rather than calendar
     day because the provider's reset time is not verified here, and a rolling
     window cannot overspend whenever the reset actually falls. Charging on
     attempt covers the undocumented case of failed calls also counting.

When the provider signals a limit - HTTP 429, or a 200 whose `errors` object
mentions one - the refresher stops for an hour rather than retrying into a
block. When the provider's own `x-ratelimit-requests-remaining` header is
present it is trusted over our count, since it also sees calls made with the
same key from anywhere else.

Configure on the *service*, never in the web build (see .env.example):

    LIVE_API_KEY=...          required to switch live scores on
    LIVE_DAILY_LIMIT=100      the plan's daily cap
    LIVE_MINUTE_LIMIT=10      the plan's per-minute cap
    LIVE_RESERVE=10           never spend the last N of the day
    LIVE_MIN_INTERVAL_S=180   floor between polls, however much budget is left
    LIVE_LEAGUES=39,140,...   optional; keep only these league ids

Storage is `DATABASE_URL` when set. Otherwise a SQLite file beside the record,
so the budget still survives a restart on one machine - the one thing an
in-memory counter cannot do. That fallback assumes a single worker; more than
one needs Postgres, whose advisory lock keeps refreshers from racing.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, func, select, text

from predictor import db

PROVIDER = "api-football"
BASE = os.environ.get("LIVE_BASE_URL",
                      "https://v3.football.api-sports.io").rstrip("/")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DAILY_LIMIT = _int("LIVE_DAILY_LIMIT", 100)
MINUTE_LIMIT = _int("LIVE_MINUTE_LIMIT", 10)
RESERVE = _int("LIVE_RESERVE", 10)
MIN_INTERVAL = _int("LIVE_MIN_INTERVAL_S", 180)
MAX_INTERVAL = _int("LIVE_MAX_INTERVAL_S", 1800)
COOLDOWN = _int("LIVE_COOLDOWN_S", 3600)

PRE_KICKOFF = timedelta(minutes=5)
MATCH_SPAN = timedelta(minutes=130)     # 90 + half-time + stoppage + margin
WINDOW = timedelta(hours=24)            # rolling budget window
TICK = 60                               # how often the refresher reconsiders
LOCK_ID = 4711_2027                     # distinct from the record's lock

STATUS_LIVE = {"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"}
STATUS_FT = {"FT", "AET", "PEN"}

# API-Football league ids per product division. The European top tens are
# confident; the deep-African ids are best guesses. A wrong id only means a
# missing chip - the frontend matches on both team names, so it can never put
# a score on the wrong fixture. Correct them with `predict.py live-leagues`.
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


def tracked_leagues() -> set:
    raw = os.environ.get("LIVE_LEAGUES", "")
    ids = {int(x) for x in raw.split(",") if x.strip().isdigit()}
    return ids or set(DIV_LEAGUES.values())


def key() -> str:
    return os.environ.get("LIVE_API_KEY", "").strip()


def configured() -> bool:
    return bool(key())


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(dt):
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ------------------------------------------------------------- transport
def http_get(url: str, headers: dict, timeout: int = 20):
    """(status, headers, body). An HTTP error is a response, not an exception:
    a 429 carries exactly the headers that say how much quota is left."""
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        hdrs = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
        try:
            body = e.read()
        except Exception:
            body = b""
        return e.code, hdrs, body


def _to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _json(raw):
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw or "{}")
    except (ValueError, AttributeError):
        return {}


def _mentions_limit(errors) -> bool:
    """Whether a provider `errors` object is a quota complaint.

    The exact wording is not something to depend on, so this errs towards
    reading an error as a limit. Backing off an hour on a false positive costs
    some freshness; hammering on a false negative risks a blocked key.
    """
    if not errors:
        return False
    if isinstance(errors, dict) and any(
            str(k).lower() in ("requests", "ratelimit", "rate_limit")
            for k in errors):
        return True
    s = json.dumps(errors).lower()
    return "limit" in s or "too many" in s


def _short(errors) -> str:
    return json.dumps(errors)[:200] if errors else ""


# ----------------------------------------------------------------- storage
_TABLES = [db.provider_calls, db.live_snapshot]


class LiveStore:
    """The budget ledger and the snapshot, over any SQLAlchemy engine."""

    def __init__(self, eng):
        self.engine = eng
        db.metadata.create_all(eng, tables=_TABLES)

    # -- ledger
    def spent_since(self, since: datetime) -> int:
        t = db.provider_calls
        q = (select(func.count()).select_from(t)
             .where(t.c.provider == PROVIDER, t.c.called_at >= since))
        with self.engine.connect() as c:
            return int(c.execute(q).scalar() or 0)

    def _latest(self, col, *where):
        t = db.provider_calls
        q = (select(col).where(t.c.provider == PROVIDER, *where)
             .order_by(t.c.called_at.desc(), t.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            return c.execute(q).scalar()

    def last_attempt_at(self):
        return self._latest(db.provider_calls.c.called_at)

    def last_rate_limited_at(self):
        t = db.provider_calls
        return self._latest(t.c.called_at, t.c.note.like("rate-limited%"))

    def provider_remaining(self, since: datetime):
        t = db.provider_calls
        return self._latest(t.c.remaining_day, t.c.remaining_day.isnot(None),
                            t.c.called_at >= since)

    def begin_call(self, endpoint: str, when: datetime) -> int:
        """Charge the attempt. Called before the request leaves the machine."""
        t = db.provider_calls
        with self.engine.begin() as c:
            r = c.execute(t.insert().values(
                provider=PROVIDER, endpoint=endpoint[:120], called_at=when,
                ok=False, note="attempting"))
            return int(r.inserted_primary_key[0])

    def finish_call(self, call_id, http_status, ok, remaining_day,
                    remaining_minute, note) -> None:
        t = db.provider_calls
        with self.engine.begin() as c:
            c.execute(t.update().where(t.c.id == call_id).values(
                http_status=http_status, ok=bool(ok),
                remaining_day=remaining_day, remaining_minute=remaining_minute,
                note=(note or "")[:240]))

    # -- snapshot
    def save_snapshot(self, matches: list, when: datetime) -> None:
        with self.engine.begin() as c:
            c.execute(db.live_snapshot.insert().values(
                provider=PROVIDER, fetched_at=when, matches=len(matches),
                body=json.dumps(matches)))

    def latest_snapshot(self):
        s = db.live_snapshot
        q = (select(s.c.id, s.c.fetched_at, s.c.body)
             .where(s.c.provider == PROVIDER)
             .order_by(s.c.fetched_at.desc(), s.c.id.desc()).limit(1))
        with self.engine.connect() as c:
            row = c.execute(q).first()
        if row is None:
            return None
        try:
            matches = json.loads(row.body)
        except ValueError:
            matches = []
        return {"id": row.id, "fetched_at": row.fetched_at, "matches": matches}

    def prune(self, when: datetime, calls_for=timedelta(days=3),
              snapshots_for=timedelta(hours=6)) -> None:
        """Old ledger rows can go once they are well outside the window; the
        latest snapshot is always kept, however old."""
        latest = self.latest_snapshot()
        t, s = db.provider_calls, db.live_snapshot
        with self.engine.begin() as c:
            c.execute(t.delete().where(t.c.called_at < when - calls_for))
            q = s.delete().where(s.c.fetched_at < when - snapshots_for)
            if latest is not None:
                q = q.where(s.c.id != latest["id"])
            c.execute(q)

    @contextmanager
    def single_flight(self):
        """One refresher at a time across every process sharing the database.

        Postgres only - SQLite is the single-machine fallback, where the
        in-process lock in `refresh` is enough.
        """
        if self.engine.dialect.name != "postgresql":
            yield True
            return
        conn = self.engine.connect()
        try:
            got = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"),
                                    {"k": LOCK_ID}).scalar())
            try:
                yield got
            finally:
                if got:
                    conn.execute(text("SELECT pg_advisory_unlock(:k)"),
                                 {"k": LOCK_ID})
        finally:
            conn.close()


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_store = None
_store_guard = threading.Lock()


def ledger_path() -> str:
    root = os.environ.get("RECORD_ROOT") or _REPO
    return os.path.join(root, "data", "record", "live.db")


def default_store() -> LiveStore:
    global _store
    with _store_guard:
        if _store is None:
            eng = db.engine()
            if eng is None:
                p = ledger_path()
                os.makedirs(os.path.dirname(p), exist_ok=True)
                eng = create_engine("sqlite:///" + p.replace("\\", "/"),
                                    future=True,
                                    connect_args={"check_same_thread": False})
            _store = LiveStore(eng)
        return _store


# ------------------------------------------------------------------ policy
def in_play(kickoffs, now) -> bool:
    now = _naive_utc(now) or utcnow()
    for k in map(_naive_utc, kickoffs or ()):
        if k is not None and k - PRE_KICKOFF <= now <= k + MATCH_SPAN:
            return True
    return False


def live_seconds_left(kickoffs, now) -> float:
    """In-play time still to come across the fixtures we carry, overlaps merged."""
    now = _naive_utc(now) or utcnow()
    spans = []
    for k in map(_naive_utc, kickoffs or ()):
        if k is None:
            continue
        start, end = max(k - PRE_KICKOFF, now), k + MATCH_SPAN
        if end > start:
            spans.append((start, end))
    spans.sort()
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += (cur_e - cur_s).total_seconds()
    return total


def interval_for(available: int, kickoffs, now) -> int:
    """Spread what is left of the budget across what is left of the play.

    A quiet midweek evening polls at the floor; a full Saturday stretches the
    interval so the budget lasts until the last final whistle rather than
    running dry at half-time of the early games.
    """
    if available <= 0:
        return MAX_INTERVAL
    iv = live_seconds_left(kickoffs, now) / available
    return int(min(MAX_INTERVAL, max(MIN_INTERVAL, iv)))


def budget(store: LiveStore, now=None) -> dict:
    now = _naive_utc(now) or utcnow()
    used = store.spent_since(now - WINDOW)
    minute = store.spent_since(now - timedelta(seconds=60))
    ours = max(0, DAILY_LIMIT - RESERVE - used)
    prov = store.provider_remaining(now - WINDOW)
    available = ours if prov is None else max(0, min(ours, prov - RESERVE))
    rl = store.last_rate_limited_at()
    until = None
    if rl is not None and now < rl + timedelta(seconds=COOLDOWN):
        until = (rl + timedelta(seconds=COOLDOWN)).isoformat() + "Z"
    return {"used_24h": used, "daily_limit": DAILY_LIMIT, "reserve": RESERVE,
            "available": available, "used_last_minute": minute,
            "minute_limit": MINUTE_LIMIT, "provider_remaining": prov,
            "cooling_down_until": until}


def decide(store: LiveStore, now, kickoffs):
    """(should_call, reason). Cheapest checks first: an idle day never touches
    the database at all."""
    if not configured():
        return False, "no LIVE_API_KEY on the service"
    now = _naive_utc(now) or utcnow()
    if not in_play(kickoffs, now):
        return False, "nothing we carry is in play"
    b = budget(store, now)
    if b["cooling_down_until"]:
        return False, ("rate-limited by the provider; cooling down until %s"
                       % b["cooling_down_until"])
    if b["available"] <= 0:
        return False, ("daily budget spent (%d used in 24h, %d held in reserve)"
                       % (b["used_24h"], RESERVE))
    if b["used_last_minute"] >= max(1, MINUTE_LIMIT - 1):
        return False, ("per-minute limit (%d in the last minute)"
                       % b["used_last_minute"])
    iv = interval_for(b["available"], kickoffs, now)
    last = store.last_attempt_at()
    if last is not None:
        ago = (now - last).total_seconds()
        if ago < iv:
            return False, "polled %ds ago; next poll in %ds" % (ago, iv - ago)
    return True, "in play; polling every %ds" % iv


# ------------------------------------------------------------------- calls
def normalise(body, leagues=None) -> list:
    """Provider fixtures -> the row shape the frontend's live chip reads."""
    out = []
    rows = body.get("response") if isinstance(body, dict) else None
    for fix in rows or []:
        try:
            fx, tm = fix["fixture"], fix["teams"]
            gl = fix.get("goals") or {}
            lg = fix.get("league") or {}
            lid = lg.get("id")
            if leagues and lid not in leagues:
                continue
            home, away = tm["home"]["name"], tm["away"]["name"]
            if not home or not away:
                continue
            st = fx.get("status") or {}
            out.append({
                "league": lid, "league_name": lg.get("name"),
                "date": fx.get("date"),
                "status": st.get("short"), "minute": st.get("elapsed"),
                "home": home, "away": away,
                "hg": gl.get("home") if gl.get("home") is not None else 0,
                "ag": gl.get("away") if gl.get("away") is not None else 0,
            })
        except (KeyError, TypeError, AttributeError):
            continue
    out.sort(key=lambda m: m["date"] or "")
    return out


_flight = threading.Lock()


def refresh(store=None, now=None, kickoffs=(), transport=None) -> dict:
    """Ask the provider for live scores - if, and only if, it is worth it."""
    now = _naive_utc(now) or utcnow()
    if not _flight.acquire(blocking=False):
        return {"called": False, "reason": "a refresh is already running"}
    try:
        store = store or default_store()
        with store.single_flight() as got:
            if not got:
                return {"called": False,
                        "reason": "another process is refreshing"}
            should, reason = decide(store, now, kickoffs)
            if not should:
                return {"called": False, "reason": reason}
            return _call(store, now, transport or http_get)
    finally:
        _flight.release()


def _headers() -> dict:
    return {"x-apisports-key": key(), "Accept": "application/json"}


def _call(store: LiveStore, now: datetime, transport) -> dict:
    endpoint = "/fixtures?" + urllib.parse.urlencode({"live": "all"})
    call_id = store.begin_call(endpoint, now)          # charged before sending
    try:
        status, headers, raw = transport(BASE + endpoint, _headers(), 20)
    except Exception as e:
        store.finish_call(call_id, None, False, None, None,
                          "network: %s" % type(e).__name__)
        return {"called": True, "ok": False,
                "reason": "network error: %s" % type(e).__name__}

    hdr = {str(k).lower(): v for k, v in (headers or {}).items()}
    rem_day = _to_int(hdr.get("x-ratelimit-requests-remaining"))
    rem_min = _to_int(hdr.get("x-ratelimit-remaining"))
    body = _json(raw)
    errors = body.get("errors") if isinstance(body, dict) else None

    if status == 429 or _mentions_limit(errors):
        store.finish_call(call_id, status, False, rem_day, rem_min,
                          "rate-limited: %s" % (_short(errors) or "HTTP %s" % status))
        return {"called": True, "ok": False,
                "reason": "rate-limited; backing off for %ds" % COOLDOWN}
    if status != 200 or errors:
        store.finish_call(call_id, status, False, rem_day, rem_min,
                          "error: %s" % (_short(errors) or "HTTP %s" % status))
        return {"called": True, "ok": False,
                "reason": "provider error (HTTP %s)" % status}

    matches = normalise(body, tracked_leagues())
    store.save_snapshot(matches, now)
    store.finish_call(call_id, status, True, rem_day, rem_min,
                      "ok: %d live" % len(matches))
    store.prune(now)
    return {"called": True, "ok": True, "matches": len(matches),
            "reason": "ok"}


# ------------------------------------------------------------------- reads
def read(store=None, now=None) -> dict:
    """What /api/live serves. Reads the snapshot; never calls the provider."""
    if not configured():
        return {"provider": PROVIDER, "enabled": False, "matches": [],
                "note": "set LIVE_API_KEY on the service to enable live scores"}
    try:
        store = store or default_store()
        now = _naive_utc(now) or utcnow()
        snap = store.latest_snapshot()
        b = budget(store, now)
    except Exception as e:
        return {"provider": PROVIDER, "enabled": True, "matches": [],
                "stale": True, "error": "%s: %s" % (type(e).__name__, e)}
    if snap is None:
        return {"provider": PROVIDER, "enabled": True, "matches": [],
                "fetched_at": None, "age_s": None, "stale": True, "budget": b,
                "note": "no snapshot yet - the provider is only asked while "
                        "a match we carry is in play"}
    age = (now - snap["fetched_at"]).total_seconds()
    return {"provider": PROVIDER, "enabled": True,
            "matches": snap["matches"],
            "fetched_at": snap["fetched_at"].isoformat() + "Z",
            "age_s": int(age),
            # After the last whistle of a window the refresher stops, so the
            # final reading would otherwise sit there as "88'" indefinitely.
            "stale": age > MAX_INTERVAL + 2 * TICK,
            "budget": b}


def status(store=None, now=None) -> dict:
    """For /api/health: is the budget healthy, and when did we last ask."""
    if not configured():
        return {"enabled": False}
    try:
        store = store or default_store()
        now = _naive_utc(now) or utcnow()
        last = store.last_attempt_at()
        snap = store.latest_snapshot()
        return {"enabled": True, "budget": budget(store, now),
                "last_attempt": last.isoformat() + "Z" if last else None,
                "snapshot_age_s": int((now - snap["fetched_at"]).total_seconds())
                if snap else None,
                "refresher": bool(_refresher and _refresher.is_alive())}
    except Exception as e:
        return {"enabled": True, "error": "%s: %s" % (type(e).__name__, e)}


# --------------------------------------------------------------- refresher
_refresher = None


def start_refresher(kickoffs_fn, tick: int = TICK):
    """One background loop per process. It reconsiders every minute, but
    `decide` is what spends: most ticks cost nothing."""
    global _refresher
    if _refresher is not None or not configured():
        return _refresher

    def loop():
        while True:
            try:
                refresh(kickoffs=kickoffs_fn())
            except Exception as e:                  # never kill the loop
                print("live refresh failed: %r" % e, flush=True)
            time.sleep(tick)

    _refresher = threading.Thread(target=loop, name="live-refresher",
                                  daemon=True)
    _refresher.start()
    return _refresher


# ------------------------------------------------------------------- admin
def find_leagues(query: str, store=None, now=None, transport=None,
                 limit: int = 8) -> list:
    """Search the provider's leagues, to correct DIV_LEAGUES / LIVE_LEAGUES.

    Admin only - deliberately not exposed over HTTP, where any visitor could
    spend the quota with it. It still goes through the ledger and still
    refuses when the budget is gone, because it spends the same requests.
    """
    store = store or default_store()
    now = _naive_utc(now) or utcnow()
    b = budget(store, now)
    if b["cooling_down_until"]:
        raise RuntimeError("provider rate limit; cooling down until %s"
                           % b["cooling_down_until"])
    if b["available"] <= 0:
        raise RuntimeError("live budget spent for the rolling 24 hours")
    endpoint = "/leagues?" + urllib.parse.urlencode({"search": query})
    call_id = store.begin_call(endpoint, now)
    status_code, headers, raw = (transport or http_get)(BASE + endpoint,
                                                        _headers(), 20)
    hdr = {str(k).lower(): v for k, v in (headers or {}).items()}
    body = _json(raw)
    errors = body.get("errors") if isinstance(body, dict) else None
    ok = status_code == 200 and not errors
    note = ("rate-limited: %s" % _short(errors)
            if (status_code == 429 or _mentions_limit(errors))
            else ("ok" if ok else "error: %s" % _short(errors)))
    store.finish_call(call_id, status_code, ok,
                      _to_int(hdr.get("x-ratelimit-requests-remaining")),
                      _to_int(hdr.get("x-ratelimit-remaining")), note)
    if not ok:
        raise RuntimeError("league search failed: HTTP %s %s"
                           % (status_code, _short(errors)))
    out = []
    for row in (body.get("response") or [])[:limit]:
        lg, c = row.get("league") or {}, row.get("country") or {}
        out.append({"id": lg.get("id"), "name": lg.get("name"),
                    "type": lg.get("type"), "country": c.get("name")})
    return out
