"""Live scores must never cost more than the free tier allows.

API-Football's free tier is 100 requests a day and 10 a minute, and exceeding
it can get the key or the server's IP temporarily blocked. The version this
replaces polled 38 leagues one request each on the user's request path: one
cold page load was 38 calls. Every test here pins one of the rules that
replaced it, using a fake transport so no test ever spends a real request.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service import live  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 9, 13, 14, 0)                        # naive UTC
IN_PLAY = [datetime(2026, 9, 13, 13, 30, tzinfo=timezone.utc)]
IDLE = [datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)]


class Fake:
    """Stands in for the network. Records every URL it is asked for."""

    def __init__(self, status=200, body=None, headers=None, raise_=None):
        self.status = status
        self.body = body if body is not None else {"errors": [], "response": []}
        self.headers = headers if headers is not None else {
            "x-ratelimit-requests-remaining": "80",
            "x-ratelimit-remaining": "9"}
        self.raise_ = raise_
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append(url)
        if self.raise_:
            raise self.raise_
        return self.status, self.headers, json.dumps(self.body).encode()


def fixture(lid, home, away, status="2H", minute=67, hg=1, ag=0):
    return {"fixture": {"date": "2026-09-13T13:00:00+00:00",
                        "status": {"short": status, "elapsed": minute}},
            "league": {"id": lid, "name": "L%d" % lid},
            "teams": {"home": {"name": home}, "away": {"name": away}},
            "goals": {"home": hg, "away": ag}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_API_KEY", "test-key-not-real")
    monkeypatch.delenv("LIVE_LEAGUES", raising=False)
    eng = create_engine("sqlite:///" + str(tmp_path / "live.db").replace("\\", "/"))
    return live.LiveStore(eng)


def charge(store, n, when):
    for _ in range(n):
        store.begin_call("/fixtures?live=all", when)


# ------------------------------------------------ the frontend never spends
def test_reading_live_scores_never_calls_the_provider(store):
    """/api/live is polled every minute by every open client. It must only
    ever read the snapshot."""
    for _ in range(50):
        live.read(store, NOW)
    assert store.spent_since(NOW - timedelta(days=1)) == 0


def test_the_web_app_has_no_route_to_the_provider():
    """The key and the provider's host must never appear in frontend code -
    anything there ships to every browser and every copy of the phone app."""
    hits = []
    for base, _, files in os.walk(os.path.join(REPO, "web", "src")):
        for f in files:
            if f.endswith((".js", ".jsx", ".ts", ".tsx")):
                src = open(os.path.join(base, f), encoding="utf-8").read()
                for needle in ("api-sports.io", "x-apisports-key"):
                    if needle in src:
                        hits.append("%s: %s" % (f, needle))
    assert hits == []


def test_no_key_means_disabled_and_nothing_spent(store, monkeypatch):
    monkeypatch.delenv("LIVE_API_KEY", raising=False)
    fake = Fake()
    assert live.read(store, NOW)["enabled"] is False
    r = live.refresh(store, NOW, IN_PLAY, fake)
    assert r["called"] is False and fake.calls == []


# ------------------------------------------------------ only when necessary
def test_nothing_in_play_means_no_call(store):
    fake = Fake()
    r = live.refresh(store, NOW, IDLE, fake)
    assert r["called"] is False and "in play" in r["reason"]
    assert fake.calls == [] and store.spent_since(NOW - timedelta(days=1)) == 0


def test_one_request_per_poll_whatever_the_number_of_leagues(store):
    """The old code sent one request per league - 38 of them."""
    fake = Fake(body={"errors": [], "response": [
        fixture(39, "Arsenal", "Chelsea"), fixture(140, "Getafe", "Celta")]})
    r = live.refresh(store, NOW, IN_PLAY, fake)
    assert r["called"] and r["ok"]
    assert len(fake.calls) == 1
    assert "live=all" in fake.calls[0] and "league=" not in fake.calls[0]


def test_the_interval_between_polls_is_respected(store):
    ok = Fake()
    assert live.refresh(store, NOW, IN_PLAY, ok)["called"]
    soon = live.refresh(store, NOW + timedelta(seconds=30), IN_PLAY, ok)
    assert soon["called"] is False and "ago" in soon["reason"]
    later = live.refresh(store, NOW + timedelta(seconds=live.MIN_INTERVAL + 1),
                         IN_PLAY, ok)
    assert later["called"]
    assert len(ok.calls) == 2


# -------------------------------------------------------------- accounting
def test_an_attempt_is_charged_before_it_is_sent(store):
    """A call that dies on the network still counts - it is not documented
    whether the provider bills failures, so we assume it does."""
    r = live.refresh(store, NOW, IN_PLAY, Fake(raise_=OSError("boom")))
    assert r["called"] and r["ok"] is False
    assert store.spent_since(NOW - timedelta(hours=1)) == 1


def test_the_daily_reserve_is_never_spent(store):
    charge(store, live.DAILY_LIMIT - live.RESERVE, NOW - timedelta(hours=2))
    fake = Fake()
    r = live.refresh(store, NOW, IN_PLAY, fake)
    assert r["called"] is False and "budget" in r["reason"]
    assert fake.calls == []


def test_the_window_is_rolling_not_a_calendar_day(store):
    """Calls more than 24 hours old free their budget, whenever the provider's
    own reset actually falls."""
    charge(store, live.DAILY_LIMIT, NOW - timedelta(hours=25))
    assert live.refresh(store, NOW, IN_PLAY, Fake())["called"]


def test_the_providers_own_count_is_trusted_over_ours(store):
    """It also sees calls made with the same key from anywhere else."""
    cid = store.begin_call("/fixtures?live=all", NOW - timedelta(minutes=10))
    store.finish_call(cid, 200, True, live.RESERVE, 9, "ok")
    r = live.refresh(store, NOW, IN_PLAY, Fake())
    assert r["called"] is False and "budget" in r["reason"]


def test_the_per_minute_limit_is_respected(store):
    charge(store, live.MINUTE_LIMIT - 1, NOW - timedelta(seconds=20))
    r = live.refresh(store, NOW, IN_PLAY, Fake())
    assert r["called"] is False and "minute" in r["reason"]


def test_the_budget_survives_a_restart(tmp_path, monkeypatch):
    """An in-memory counter forgets how much of the day is spent."""
    monkeypatch.setenv("LIVE_API_KEY", "k")
    dsn = "sqlite:///" + str(tmp_path / "live.db").replace("\\", "/")
    live.LiveStore(create_engine(dsn)).begin_call("/fixtures?live=all", NOW)
    fresh = live.LiveStore(create_engine(dsn))         # a new process
    assert fresh.spent_since(NOW - timedelta(hours=1)) == 1


# ---------------------------------------------------------- rate limiting
def test_a_429_backs_off_instead_of_retrying(store):
    """Retrying into a limit is how a key gets blocked."""
    r = live.refresh(store, NOW, IN_PLAY,
                     Fake(status=429, body={"errors": {"requests": "too many"}}))
    assert r["called"] and "rate-limited" in r["reason"]
    ok = Fake()
    blocked = live.refresh(store, NOW + timedelta(minutes=10), IN_PLAY, ok)
    assert blocked["called"] is False and "cooling down" in blocked["reason"]
    assert ok.calls == []
    after = live.refresh(store, NOW + timedelta(seconds=live.COOLDOWN + 1),
                         [NOW.replace(tzinfo=timezone.utc) + timedelta(minutes=50)],
                         ok)
    assert after["called"]


def test_a_200_carrying_a_limit_error_is_still_a_limit(store):
    body = {"errors": {"requests": "You have reached the request limit for the day"},
            "response": []}
    r = live.refresh(store, NOW, IN_PLAY, Fake(status=200, body=body))
    assert "rate-limited" in r["reason"]
    assert store.last_rate_limited_at() is not None


# --------------------------------------------------------------- snapshot
def test_the_snapshot_keeps_the_row_shape_the_chip_reads(store):
    body = {"errors": [], "response": [
        fixture(39, "Arsenal", "Chelsea", hg=2, ag=1),
        fixture(99999, "Somewhere", "Else")]}          # a league we do not carry
    live.refresh(store, NOW, IN_PLAY, Fake(body=body))
    got = live.read(store, NOW + timedelta(seconds=5))
    assert got["enabled"] and got["stale"] is False
    assert [m["home"] for m in got["matches"]] == ["Arsenal"]
    m = got["matches"][0]
    assert {"home", "away", "status", "minute", "hg", "ag"} <= set(m)
    assert (m["hg"], m["ag"], m["status"]) == (2, 1, "2H")
    assert got["budget"]["provider_remaining"] == 80


def test_a_frozen_snapshot_is_marked_stale(store):
    """After the last whistle the refresher stops; an old '88 minutes' must
    not keep showing as live."""
    store.save_snapshot([{"home": "A", "away": "B", "status": "2H"}],
                        NOW - timedelta(seconds=live.MAX_INTERVAL + 300))
    assert live.read(store, NOW)["stale"] is True


# ------------------------------------------------------------------ admin
def test_league_search_is_charged_like_any_other_call(store):
    body = {"errors": [], "response": [
        {"league": {"id": 509, "name": "Ligi Kuu Bara", "type": "League"},
         "country": {"name": "Tanzania"}}]}
    got = live.find_leagues("tanzania", store=store, now=NOW,
                            transport=Fake(body=body))
    assert got[0]["id"] == 509
    assert store.spent_since(NOW - timedelta(hours=1)) == 1


def test_league_search_refuses_when_the_budget_is_gone(store):
    charge(store, live.DAILY_LIMIT, NOW - timedelta(hours=1))
    fake = Fake()
    with pytest.raises(RuntimeError):
        live.find_leagues("tanzania", store=store, now=NOW, transport=fake)
    assert fake.calls == []
