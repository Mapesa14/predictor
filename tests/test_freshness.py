"""Does the product notice when its own data stops arriving?

Written after the failure it exists to prevent: on 2026-10-09 the fixtures feed
had not been downloaded since 12 September and the Tanzanian scraper had been
crashing for three weeks. Every endpoint answered 200 and the screen showed an
empty day, which a reader cannot tell apart from "no matches today".
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import freshness  # noqa: E402
from service import alerts  # noqa: E402

NOW = datetime(2026, 10, 9, 6, 0, tzinfo=timezone.utc)      # 09:00 EAT


def age(path, hours):
    t = (NOW - timedelta(hours=hours)).timestamp()
    os.utime(path, (t, t))


def tree(tmp_path, days_ahead=(0, 1), hours_old=1.0):
    """A data folder and a repo, with a schedule covering the given days."""
    data, repo = tmp_path / "soccer", tmp_path / "repo"
    (repo / "data" / "manual" / "fixtures").mkdir(parents=True, exist_ok=True)
    (repo / "data" / "record").mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)
    rows = ["Div,Date,Time,HomeTeam,AwayTeam"]
    for d in days_ahead:
        day = (NOW + timedelta(days=d)).strftime("%d/%m/%Y")
        rows.append("E0,%s,20:00,Arsenal,Leeds" % day)
    (data / "fixtures.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data / "E0.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,01/10/2026,A,B,1,0\n",
        encoding="utf-8")
    for name in ("TZ1.csv", "fixtures_api.csv", "results_api.csv"):
        (repo / "data" / "manual" / name).write_text("Div\nE0\n", encoding="utf-8")
    (repo / "data" / "record" / "predictions.csv").write_text("x\n", encoding="utf-8")
    for p in [data / "fixtures.csv", data / "E0.csv",
              repo / "data" / "record" / "predictions.csv"] + \
             list((repo / "data" / "manual").glob("*.csv")):
        age(p, hours_old)
    return data, repo


def report(tmp_path, **kw):
    data, repo = tree(tmp_path, **kw)
    return freshness.report(str(data), str(repo), str(repo), now=NOW)


def level(rep, key):
    return next(s["level"] for s in rep["sources"] if s["key"] == key)


# --------------------------------------------------------- what a reader sees
def test_a_schedule_with_tomorrow_on_it_is_healthy(tmp_path):
    rep = report(tmp_path)
    sched = next(s for s in rep["sources"] if s["key"] == "schedule")
    assert (sched["level"], sched["today"], sched["tomorrow"]) == ("ok", 1, 1)
    assert rep["overall"] == "ok" and not rep["problems"]


def test_a_board_that_ran_out_weeks_ago_is_stale(tmp_path):
    """The failure of 2026-10-09: the newest fixture on file was three weeks
    old, and the screen drew it as an ordinary empty day."""
    rep = report(tmp_path, days_ahead=(-21, -20))
    sched = next(s for s in rep["sources"] if s["key"] == "schedule")
    assert sched["level"] == "stale" and (sched["today"], sched["tomorrow"]) == (0, 0)
    assert "ends" in sched["note"]
    assert "schedule" in rep["problems"] and rep["overall"] == "stale"


def test_a_board_that_ends_today_warns(tmp_path):
    rep = report(tmp_path, days_ahead=(0,))
    assert level(rep, "schedule") == "warn"


def test_an_empty_schedule_is_stale(tmp_path):
    rep = report(tmp_path, days_ahead=())
    assert level(rep, "schedule") == "stale"


def test_a_quiet_day_inside_a_covered_window_is_not_a_fault(tmp_path):
    """Nothing tomorrow, matches the day after: an international break, not a
    broken pipeline."""
    rep = report(tmp_path, days_ahead=(0, 2, 3))
    sched = next(s for s in rep["sources"] if s["key"] == "schedule")
    assert sched["tomorrow"] == 0 and sched["level"] == "ok"
    assert rep["overall"] == "ok"


# ------------------------------------------------------------- refresh ages
def test_a_source_that_stopped_arriving_warns_then_goes_stale(tmp_path):
    fresh = report(tmp_path, hours_old=1)
    assert level(fresh, "feed") == "ok"
    warn = report(tmp_path, hours_old=freshness.AGE_WARN + 1)
    assert level(warn, "feed") == "warn"
    dead = report(tmp_path, hours_old=freshness.AGE_STALE + 1)
    assert level(dead, "feed") == "stale" and dead["overall"] == "stale"


def test_the_api_sources_are_held_to_the_six_hourly_cycle(tmp_path):
    """They are fetched every refresh, so a day old is already a missed cycle."""
    rep = report(tmp_path, hours_old=freshness.API_AGE_WARN + 1)
    assert level(rep, "api_fixtures") == "warn" and level(rep, "feed") == "ok"


def test_staleness_is_judged_on_the_refresh_not_the_last_match(tmp_path):
    """An international break is not a fault: no league played for a fortnight
    in September, and every file was arriving on time."""
    data, repo = tree(tmp_path)
    (data / "E0.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,01/08/2026,A,B,1,0\n",
        encoding="utf-8")
    age(data / "E0.csv", 1)
    rep = freshness.report(str(data), str(repo), str(repo), now=NOW)
    assert rep["overall"] == "ok"


def test_a_missing_source_is_stale_not_invisible(tmp_path):
    data, repo = tree(tmp_path)
    os.remove(repo / "data" / "manual" / "TZ1.csv")
    rep = freshness.report(str(data), str(repo), str(repo), now=NOW)
    assert level(rep, "tanzania") == "stale"
    assert next(s for s in rep["sources"] if s["key"] == "tanzania")["last"] == "never"


# ----------------------------------------------------------------- the tips
def test_todays_lists_are_only_due_after_the_freeze_hour(tmp_path):
    data, repo = tree(tmp_path)
    early = NOW.replace(hour=3)                     # 06:00 EAT, before 10:00
    assert level(freshness.report(str(data), str(repo), str(repo), now=early),
                 "tips") == "ok"
    late = NOW.replace(hour=12)                     # 15:00 EAT
    assert level(freshness.report(str(data), str(repo), str(repo), now=late),
                 "tips") == "stale"


# ------------------------------------------------------------------ headline
def test_the_headline_names_what_broke(tmp_path):
    rep = report(tmp_path, days_ahead=(-21,))
    line = freshness.headline(rep)
    assert "Fixtures on the board" in line and "stale" in line
    assert "all sources current" in freshness.headline(report(tmp_path))


# ------------------------------------------------------------------- alerts
class Hook:
    def __init__(self, status=200):
        self.status, self.sent = status, []

    def __call__(self, url, text):
        self.sent.append((url, text))
        return self.status


@pytest.fixture
def hooked(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example/abc")
    (tmp_path / "data" / "record").mkdir(parents=True)
    return str(tmp_path)


def test_an_alert_is_posted_once_then_held_back(hooked):
    h = Hook()
    first = alerts.notify(hooked, "source:feed", "feed is stale", NOW, h)
    again = alerts.notify(hooked, "source:feed", "feed is stale",
                          NOW + timedelta(hours=1), h)
    assert first["sent"] and not again["sent"] and len(h.sent) == 1
    later = alerts.notify(hooked, "source:feed", "feed is stale",
                          NOW + timedelta(hours=alerts.REPEAT_HOURS + 1), h)
    assert later["sent"] and len(h.sent) == 2
    assert json.loads(json.dumps(h.sent[0][1])).startswith("feed")


def test_a_problem_that_clears_is_announced_again(hooked):
    h = Hook()
    alerts.notify(hooked, "source:feed", "feed is stale", NOW, h)
    alerts.clear(hooked, ["source:feed"])
    r = alerts.notify(hooked, "source:feed", "feed is stale",
                      NOW + timedelta(minutes=5), h)
    assert r["sent"] and len(h.sent) == 2


def test_without_a_webhook_nothing_is_sent_and_nothing_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    (tmp_path / "data" / "record").mkdir(parents=True)
    r = alerts.notify(str(tmp_path), "source:feed", "feed is stale", NOW)
    assert not r["sent"] and "ALERT_WEBHOOK_URL" in r["reason"]


def test_a_refused_webhook_is_reported_not_raised(hooked):
    r = alerts.notify(hooked, "source:feed", "x", NOW, Hook(status=500))
    assert not r["sent"] and "500" in r["reason"]


def test_check_alerts_on_bad_sources_and_failed_steps(hooked, tmp_path):
    rep = {"sources": [{"key": "feed", "label": "Feed", "level": "stale",
                        "last": "9 days ago", "note": "", "fix": "run it"},
                       {"key": "tips", "label": "Tips", "level": "ok",
                        "last": "today", "note": "", "fix": ""}]}
    h = Hook()
    out = alerts.check(hooked, rep, ["tanzania failed: KeyError"], NOW, h)
    assert out["sent"] == 2
    assert any("Feed is stale" in t for _, t in h.sent)
    assert any("tanzania failed" in t for _, t in h.sent)


def test_a_reader_is_not_warned_about_background_tasks(tmp_path):
    """The record falling behind is an operational fault, not a reason to
    distrust today's slate - and a banner that cries wolf is ignored."""
    data, repo = tree(tmp_path)
    os.remove(repo / "data" / "record" / "predictions.csv")
    rep = freshness.report(str(data), str(repo), str(repo), now=NOW)
    assert rep["data_level"] == "ok"
    assert rep["publishing_level"] == "stale" and rep["overall"] == "stale"
