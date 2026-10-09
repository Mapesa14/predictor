"""Frozen tip lists and their results.

What these defend: the lists judged are the lists that were shown, frozen
before kick-off and never changed; a result is never stored, so it cannot be
quietly corrected; and the numbers shown cannot flatter the lists - void
matches are left out, pending ones are not counted as won, and the official
score beats the fallback source.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import tipslog  # noqa: E402
from service import fixtures_api as fa  # noqa: E402
from tests.test_db import sqlite_db  # noqa: E402,F401
from tests.test_fixtures_api import NOW as API_NOW, fx, p, store  # noqa: E402,F401
from tests.test_live import Fake  # noqa: E402

NOW = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)       # 09:00 EAT
TODAY = "2026-09-13T18:00:00+03:00"


def row(home, away, p1, div="E0", ko=TODAY, odds=None, market="same"):
    px, p2 = round((1 - p1) * 0.6, 4), round((1 - p1) * 0.4, 4)
    m = {"div": div, "league": "League " + div, "date": ko, "home": home,
         "away": away, "p": {"1": p1, "X": px, "2": p2}, "score": "1-0"}
    m["market"] = {"1": p1, "X": px, "2": p2} if market == "same" else market
    if odds:
        m["odds"] = {"1": odds, "X": 4.0, "2": 6.0}
    return m


def official(*rows):
    df = pd.DataFrame(list(rows), columns=["Div", "Date", "HomeTeam", "AwayTeam",
                                           "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def api(*rows):
    return pd.DataFrame(list(rows), columns=["Div", "Date", "HomeTeam", "AwayTeam",
                                             "FTHG", "FTAG", "Status"])


def frozen(tmp_path, rows):
    tipslog.freeze(rows, str(tmp_path), now=NOW)
    return tipslog.load(str(tmp_path))


# ----------------------------------------------------------------- freeze
def test_a_days_lists_are_frozen_once(tmp_path):
    rows = [row("Arsenal", "Leeds", 0.82, odds=1.3), row("Spurs", "Wolves", 0.70)]
    r = tipslog.freeze(rows, str(tmp_path), now=NOW)
    assert r["frozen"] and r["counts"]["bankers"] == 1 and r["counts"]["long_list"] == 2
    again = tipslog.freeze([row("Chelsea", "Fulham", 0.90)], str(tmp_path), now=NOW)
    assert not again["frozen"]
    log = tipslog.load(str(tmp_path))
    assert "Chelsea" not in set(log["home"])
    assert (log["list"] == tipslog.DAY_MARK).sum() == 1


def test_only_todays_fixtures_not_yet_started_are_eligible(tmp_path):
    rows = [row("Arsenal", "Leeds", 0.82),
            row("Spurs", "Wolves", 0.82, ko="2026-09-14T18:00:00+03:00"),   # tomorrow
            row("Derby", "Hull", 0.82, ko="2026-09-13T08:00:00+03:00")]     # under way
    log = frozen(tmp_path, rows)
    assert set(log.loc[log["list"] == "bankers", "home"]) == {"Arsenal"}


def test_an_empty_day_is_still_recorded_as_frozen(tmp_path):
    r = tipslog.freeze([], str(tmp_path), now=NOW)
    assert r["frozen"] and r["written"] == 0
    assert tipslog.is_frozen(str(tmp_path), NOW)
    assert not tipslog.freeze([row("Arsenal", "Leeds", 0.9)], str(tmp_path), now=NOW)["frozen"]


def test_a_changed_pick_breaks_the_chain(tmp_path):
    frozen(tmp_path, [row("Arsenal", "Leeds", 0.82), row("Spurs", "Wolves", 0.70)])
    assert tipslog.verify(str(tmp_path))["ok"]
    f = tipslog.path(str(tmp_path))
    df = pd.read_csv(f, dtype=str)
    df.loc[df["home"] == "Spurs", "p"] = "0.99"
    df.to_csv(f, index=False)
    assert not tipslog.verify(str(tmp_path))["ok"]


def test_the_price_on_the_pick_is_kept(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82, odds=1.3)])
    pick = log[log["list"] == "bankers"].iloc[0]
    assert pick["odds"] == 1.3 and pick["pick"] == "1" and pick["side"] == "Arsenal"


# -------------------------------------------------------------- settling
def test_official_results_settle_the_picks(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82), row("Spurs", "Wolves", 0.80)])
    s = tipslog.settle(log, official(("E0", "2026-09-13", "Arsenal", "Leeds", 2, 0),
                                     ("E0", "2026-09-13", "Spurs", "Wolves", 1, 1)))
    got = dict(zip(s["home"], zip(s["status"], s["dc_won"])))
    assert got["Arsenal"] == ("won", True)
    assert got["Spurs"] == ("lost", True)            # a draw loses the pick, not the double chance


def test_an_api_score_fills_what_the_official_sources_lack(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82)])
    s = tipslog.settle(log, official(),
                       api(("E0", "2026-09-13", "Arsenal", "Leeds", 0, 1, "FT")))
    b = s[s["list"] == "bankers"].iloc[0]
    assert (b["status"], b["source"], b["score"]) == ("lost", "api", "0-1")


def test_the_official_score_wins_and_the_disagreement_is_flagged(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82)])
    s = tipslog.settle(log, official(("E0", "2026-09-13", "Arsenal", "Leeds", 2, 1)),
                       api(("E0", "2026-09-13", "Arsenal", "Leeds", 1, 1, "FT")))
    b = s[s["list"] == "bankers"].iloc[0]
    assert (b["status"], b["source"], b["conflict"]) == ("won", "official", True)


def test_a_postponed_match_is_void_and_left_out_of_the_counts(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82), row("Spurs", "Wolves", 0.80)])
    s = tipslog.settle(log, official(("E0", "2026-09-13", "Spurs", "Wolves", 3, 0)),
                       api(("E0", "2026-09-13", "Arsenal", "Leeds", None, None, "PST")))
    st = tipslog.stats(s[s["list"] == "bankers"])
    assert (st["settled"], st["won"], st["void"], st["hit"]) == (1, 1, 1, 1.0)


def test_no_result_yet_is_pending_never_won(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82), row("Spurs", "Wolves", 0.80)])
    s = tipslog.settle(log, official(("E0", "2026-09-13", "Spurs", "Wolves", 3, 0)))
    bk = s[s["list"] == "bankers"]
    st = tipslog.stats(bk)
    assert (st["settled"], st["pending"]) == (1, 1)
    assert tipslog._acca(bk)["all_won"] is None         # undecided, not "won"


def test_one_losing_banker_decides_the_accumulator(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82), row("Spurs", "Wolves", 0.80)])
    s = tipslog.settle(log, official(("E0", "2026-09-13", "Spurs", "Wolves", 0, 2)))
    acca = tipslog._acca(s[s["list"] == "bankers"])
    assert acca["all_won"] is False and acca["legs"] == 2
    assert acca["expected"] == pytest.approx(0.82 * 0.80)


def test_a_late_kick_off_dated_a_day_later_at_source_still_settles(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82)])
    s = tipslog.settle(log, official(("E0", "2026-09-14", "Arsenal", "Leeds", 1, 0)))
    assert s[s["list"] == "bankers"].iloc[0]["status"] == "won"


def test_flat_return_is_taken_at_the_frozen_odds(tmp_path):
    log = frozen(tmp_path, [row("Arsenal", "Leeds", 0.82, odds=1.5),
                            row("Spurs", "Wolves", 0.80, odds=1.4),
                            row("Derby", "Hull", 0.79)])              # no price: not staked
    s = tipslog.settle(log, official(("E0", "2026-09-13", "Arsenal", "Leeds", 1, 0),
                                     ("E0", "2026-09-13", "Spurs", "Wolves", 0, 0),
                                     ("E0", "2026-09-13", "Derby", "Hull", 2, 0)))
    st = tipslog.stats(s[s["list"] == "bankers"])
    assert st["priced"] == 2
    assert st["flat_return"] == pytest.approx((0.5 - 1.0) / 2)
    assert st["expected_hit"] == pytest.approx((0.82 + 0.80 + 0.79) / 3)


def test_the_summary_carries_days_totals_and_what_was_promised(tmp_path):
    frozen(tmp_path, [row("Arsenal", "Leeds", 0.82), row("Spurs", "Wolves", 0.70)])
    s = tipslog.summary(str(tmp_path),
                        official(("E0", "2026-09-13", "Arsenal", "Leeds", 2, 0)),
                        now=NOW)
    assert s["frozen_days"] == 1 and s["chain"]["ok"]
    day = s["days"][0]
    assert day["lists"]["bankers"]["stats"]["won"] == 1
    assert day["lists"]["long_list"]["stats"]["pending"] == 1
    assert s["totals"]["bankers"]["all"]["settled"] == 1
    assert s["totals"]["bankers"]["promised_hit"] == tipslog.PROMISED["bankers"]
    assert not s["totals"]["bankers"]["all"]["enough"]


def test_the_database_store_holds_the_same_lists(tmp_path, sqlite_db):  # noqa: F811
    r = tipslog.freeze([row("Arsenal", "Leeds", 0.82, odds=1.3)], str(tmp_path), now=NOW)
    assert r["frozen"] and r["backend"].startswith("db:")
    assert tipslog.verify(str(tmp_path))["ok"]
    assert not tipslog.freeze([row("Spurs", "Wolves", 0.9)], str(tmp_path), now=NOW)["frozen"]
    assert not os.path.exists(tipslog.path(str(tmp_path)))


# ------------------------------------------------ API-Football final scores
def finished(lid, home, away, hg, ag, status="FT"):
    f = fx(lid, home, away, status=status)
    f["goals"] = {"home": hg, "away": ag}
    f["score"] = {"fulltime": {"home": hg, "away": ag}}
    return f


def test_sync_keeps_final_scores_and_voids(tmp_path, p, store):  # noqa: F811
    body = {"errors": [], "response": [
        finished(39, "Liverpool", "Tottenham", 2, 1),
        finished(140, "Rayo Vallecano", "Espanyol", None, None, status="PST")]}
    r = fa.sync(str(tmp_path), p, days=1, now=API_NOW, store=store,
                transport=Fake(body=body))
    assert r["results"] == 2
    res = tipslog.api_results(str(tmp_path))
    got = {(h, st): (hg, ag) for h, st, hg, ag in
           res[["HomeTeam", "Status", "FTHG", "FTAG"]].itertuples(index=False)}
    assert got[("Liverpool", "FT")] == (2, 1)
    assert ("Vallecano", "PST") in got
    fixtures = pd.read_csv(fa.api_file(str(tmp_path)))
    assert "Vallecano" not in set(fixtures["HomeTeam"])        # void: not a fixture


def test_yesterday_is_fetched_once_a_day(tmp_path, p, store):  # noqa: F811
    body = {"errors": [], "response": []}
    f1 = Fake(body=body)
    fa.sync(str(tmp_path), p, days=1, now=API_NOW, store=store, transport=f1,
            yesterday=True)
    assert len(f1.calls) == 2 and "date=2026-09-14" in f1.calls[0]
    f2 = Fake(body=body)
    fa.sync(str(tmp_path), p, days=1, now=API_NOW.replace(hour=12), store=store,
            transport=f2, yesterday=True)
    assert len(f2.calls) == 1
