"""European cup fixtures from BBC Sport.

This source exists because of a real gap: football-data has never carried a
Champions League tie, and when the API-Football account was suspended the whole
European programme disappeared from the product on a night with Manchester City
v Paris Saint-Germain on the board.

What these tests defend: only the competitions nothing else covers are taken
(two sources for one fixture is how a schedule contradicts itself), a club is
matched or reported but never guessed, and a failed fetch leaves the schedule
exactly as it was.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service import bbc_fixtures as bbc  # noqa: E402
from tests.test_fixtures_api import p  # noqa: E402,F401


def head(slug):
    return ('<div class="ssrcss-HeadingWrapper"><h2 class="ssrcss-GroupHeader">'
            '<a href="/sport/football/%s/table">x</a></h2></div>' % slug)


def game(home, away, t):
    return ('<span class="visually-hidden">%s versus %s kick off %s</span>'
            % (home, away, t))


PAGE = ("<html><body>"
        + head("champions-league")
        + game("Manchester City", "Real Madrid", "20:00")
        + game("Tottenham", "Liverpool", "17:45")
        + head("europa-league")
        + game("Vallecano", "Espanyol", "20:00")
        + head("championship")                       # the feed covers this one
        + game("Reading", "Barnsley", "19:45")
        + "</body></html>")


# ------------------------------------------------------------------ parsing
def test_only_the_competitions_nothing_else_covers_are_taken():
    got = bbc.parse(PAGE)
    assert {g["comp"] for g in got} == {"Champions League", "Europa League"}
    assert all(g["home"] != "Reading" for g in got)   # Championship: the feed's


def test_each_fixture_keeps_its_clubs_and_kick_off():
    g = bbc.parse(PAGE)[0]
    assert (g["home"], g["away"], g["time"]) == ("Manchester City", "Real Madrid", "20:00")


def test_html_entities_and_odd_spacing_are_cleaned():
    page = head("champions-league") + game("Bod&#xF8; /  Glimt", "Bayern&nbsp;Munich", "9:30")
    g = bbc.parse(page)[0]
    assert g["home"] == "Bodø / Glimt" and g["away"] == "Bayern Munich"
    assert g["time"] == "09:30"                       # padded, as the files expect


def test_a_page_with_nothing_we_take_yields_nothing():
    assert bbc.parse(head("championship") + game("Reading", "Barnsley", "19:45")) == []


# ------------------------------------------------------------------- rows
def test_a_tie_carries_both_clubs_leagues_for_the_bridge(p):  # noqa: F811
    p.bridge = {"offsets": {"E0": 0.45, "SP1": 0.24}}
    rows, rep = bbc.build_rows(bbc.parse(PAGE), p, "2026-10-14")
    by_home = {r["HomeTeam"]: r for r in rows}
    # the point of the bridge: two clubs, two leagues, one tie
    city = by_home["Man City"]
    assert (city["Div"], city["AwayDiv"], city["Comp"]) == ("E0", "SP1", "Champions League")
    both_english = by_home["Tottenham"]
    assert (both_english["Div"], both_english["AwayDiv"]) == ("E0", "E0")
    esp = by_home["Vallecano"]
    assert (esp["Div"], esp["AwayDiv"], esp["Comp"]) == ("SP1", "SP1", "Europa League")
    assert esp["Date"] == "14/10/2026" and esp["Time"] == "20:00"
    assert rep["unresolved"] == []


def test_a_club_we_do_not_carry_is_reported_never_guessed(p):  # noqa: F811
    p.bridge = {"offsets": {"E0": 0.45}}
    page = head("champions-league") + game("Liverpool", "Slovan Bratislava", "20:00")
    rows, rep = bbc.build_rows(bbc.parse(page), p, "2026-10-14")
    assert rows == []
    assert any("Slovan Bratislava" in u for u in rep["unresolved"])


# ------------------------------------------------------------------- sync
def test_sync_writes_the_file(tmp_path, p):  # noqa: F811
    p.bridge = {"offsets": {"E0": 0.45, "SP1": 0.24}}
    r = bbc.sync(str(tmp_path), p, days=1, now=pd.Timestamp("2026-10-14"),
                 transport=lambda url: PAGE)
    assert r["written"] and r["rows"] == 3
    got = pd.read_csv(bbc.file(str(tmp_path)))
    assert set(got["Comp"]) == {"Champions League", "Europa League"}
    assert set(got.columns) == set(bbc.COLUMNS)


def test_a_failed_fetch_never_empties_the_schedule(tmp_path, p):  # noqa: F811
    p.bridge = {"offsets": {"E0": 0.45, "SP1": 0.24}}
    bbc.sync(str(tmp_path), p, days=1, now=pd.Timestamp("2026-10-14"),
             transport=lambda url: PAGE)

    def boom(url):
        raise RuntimeError("BBC answered HTTP 503")

    r = bbc.sync(str(tmp_path), p, days=1, now=pd.Timestamp("2026-10-14"),
                 transport=boom)
    assert r["fetched"] == [] and r["errors"] and not r["written"]
    assert len(pd.read_csv(bbc.file(str(tmp_path)))) == 3


def test_resyncing_the_same_day_does_not_duplicate(tmp_path, p):  # noqa: F811
    p.bridge = {"offsets": {"E0": 0.45, "SP1": 0.24}}
    for _ in range(2):
        bbc.sync(str(tmp_path), p, days=1, now=pd.Timestamp("2026-10-14"),
                 transport=lambda url: PAGE)
    assert len(pd.read_csv(bbc.file(str(tmp_path)))) == 3


def test_the_schedule_prefers_the_feed_over_this_filler(tmp_path):
    """BBC has no closing price; football-data does. Same fixture, feed wins."""
    from predictor import fixtures
    manual = tmp_path / "manual"
    (manual / "fixtures").mkdir(parents=True)
    (tmp_path / "fixtures.csv").write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n"
        "E0,14/10/2026,20:00,Liverpool,Arsenal,2.10,3.40,3.30\n", encoding="utf-8")
    (manual / "fixtures_bbc.csv").write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,Comp,ScaleAlt,AwayDiv\n"
        "E0,14/10/2026,20:00,Liverpool,Arsenal,Champions League,,E0\n"
        "SP1,14/10/2026,21:00,Real Madrid,Vallecano,Champions League,,SP1\n",
        encoding="utf-8")
    fx = fixtures.load_any(None, str(tmp_path / "fixtures.csv"),
                           overlay_dir=str(manual / "fixtures"))
    liv = fx[fx["HomeTeam"] == "Liverpool"]
    assert len(liv) == 1 and float(liv.iloc[0]["AvgH"]) == 2.10
    assert len(fx[fx["HomeTeam"] == "Real Madrid"]) == 1      # the gap, filled


# ---------------------------------------------------------------- results
DONE = (head("champions-league")
        + '<span class="visually-hidden">Como 4 , Manchester United 1 at Full time</span>'
        + '<span class="visually-hidden">Tottenham 0 , Liverpool 2 at Full time</span>'
        + head("league-cup")      # a competition we do not take
        + '<span class="visually-hidden">Reading 1 , Barnsley 0 at Full time</span>')


def test_a_finished_tie_is_read_as_a_score():
    got = bbc.parse_results(DONE)
    assert {(g["home"], g["hg"], g["away"], g["ag"]) for g in got} == {
        ("Como", 4, "Manchester United", 1), ("Tottenham", 0, "Liverpool", 2)}


def test_extra_time_is_not_settled_as_a_ninety_minute_score():
    """"After extra time" carries the aggregate; a 90-minute selection settled
    on it would be quietly wrong."""
    page = (head("champions-league")
            + '<span class="visually-hidden">Warrenpoint 5 , Crusaders 2 After extra time</span>')
    assert bbc.parse_results(page) == []


def test_results_from_competitions_we_do_not_take_are_ignored():
    assert all(g["home"] != "Reading" for g in bbc.parse_results(DONE))


def test_results_are_written_with_the_clubs_we_know(tmp_path, p):  # noqa: F811
    p.bridge = {"offsets": {"E0": 0.45, "SP1": 0.24}}
    r = bbc.sync(str(tmp_path), p, days=1, back=1,
                 now=pd.Timestamp("2026-10-23"), transport=lambda url: DONE)
    assert r["results"] >= 1
    got = pd.read_csv(bbc.results_file(str(tmp_path)))
    row = got[got["HomeTeam"] == "Tottenham"].iloc[0]
    assert (int(row["FTHG"]), int(row["FTAG"]), row["Status"]) == (0, 2, "FT")
    assert row["Comp"] == "Champions League"


def test_a_european_pick_is_settled_from_these_results(tmp_path, p):  # noqa: F811
    """The whole point: a European tie now earns a win or a loss, not a
    permanent "not yet measured"."""
    from datetime import datetime, timezone
    from predictor import markets, model, tips, tipslog
    p.bridge = {"offsets": {"E0": 0.45}}
    now = datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc)
    s = markets.summary(model.score_matrix_from_rates(2.4, 0.6, -0.05))
    tie = {"div": "E0", "league": "Champions League", "comp": "Champions League",
           "date": "2026-10-23T22:00:00+03:00", "home": "Tottenham",
           "away": "Liverpool", "p": {"1": s["result"]["H"], "X": s["result"]["D"],
                                      "2": s["result"]["A"]},
           "market": None, "markets": tips.selections(s)}
    tipslog.freeze([tie], str(tmp_path), now=now)
    bbc.sync(str(tmp_path), p, days=1, back=1, now=pd.Timestamp("2026-10-23"),
             transport=lambda url: DONE)
    settled = tipslog.settle(tipslog.load(str(tmp_path)),
                             pd.DataFrame(columns=["Div", "Date", "HomeTeam",
                                                   "AwayTeam", "FTHG", "FTAG"]),
                             tipslog.api_results(str(tmp_path)))
    row = settled[settled["list"] == tipslog.EUROPE].iloc[0]
    assert row["status"] in ("won", "lost") and row["score"] == "0-2"
    assert row["source"] == "api"        # a fetched final, not an official one
