"""The Tanzanian source: the one league with no feed behind it.

These cover the two things that have actually gone wrong with the home league -
its results being wiped by a rebuild, and its fixtures never having dates -
plus the one thing that must never happen, a club name being guessed at.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import fixtures, tanzania  # noqa: E402


def event(eid, when, home, away, res, league="NBC PREMIER LEAGUE 2026/2027"):
    """One SportsPress event block, the shape ligikuu.co.tz publishes."""
    spans = "".join('<span class="sp-result ">%s</span>' % r for r in res)
    return (
        "<td>"
        '<span class="team-logo logo-odd" title="%s"></span>'
        '<span class="team-logo logo-even" title="%s"></span>'
        '<time class="sp-event-date" datetime="%s"><a href="/event/%s/">d</a></time>'
        '<h5 class="sp-event-results">%s</h5>'
        '<div class="sp-event-league">%s</div>'
        '<h4 class="sp-event-title"><a href="/event/%s/">%s vs %s</a></h4>'
        "</td>" % (home, away, when, eid, spans, league, eid, home, away)
    )


PAGE = "<table>" + "".join([
    event("1", "2026-09-12 16:00:00", "Polisi Tanzania", "Mbeya City", ["2", "0"]),
    event("2", "2026-09-12 19:00:00", "Coastal union", "JKT Tanzania", ["2", "1"]),
    event("3", "2026-09-13 16:00:00", "pamba Jiji", "TRA United", ["4:00 pm"]),
    event("4", "2026-09-13 19:00:00", "Kagera Sugar", "Fountain Gate", ["7:00 pm"]),
    # the same event repeated by a second widget, as the homepage does
    event("1", "2026-09-12 16:00:00", "Polisi Tanzania", "Mbeya City", ["2", "0"]),
    # a different competition on the same page
    event("9", "2026-09-13 16:00:00", "Bigman FC", "Gunners FC", ["4:00 pm"],
          league="NBC CHAMPIONSHIP LEAGUE 2026/2027"),
]) + "</table>"


# ------------------------------------------------------------------ parsing
def test_parse_splits_played_from_scheduled():
    df = tanzania.parse(PAGE)
    assert len(df) == 4                      # deduped, and second tier excluded
    played = df[df["FTHG"].notna()]
    assert len(played) == 2
    assert set(df[df["FTHG"].isna()]["Time"]) == {"16:00", "19:00"}


def test_parse_keeps_the_kick_off_time():
    df = tanzania.parse(PAGE)
    row = df[df["HomeTeam"] == "Pamba Jiji FC"].iloc[0]
    assert row["Time"] == "16:00"            # EAT, as published
    assert str(row["Date"].date()) == "2026-09-13"


def test_parse_ignores_the_second_tier():
    df = tanzania.parse(PAGE)
    assert "Bigman FC" not in set(df["raw_home"])
    other = tanzania.parse(PAGE, league="NBC CHAMPIONSHIP LEAGUE")
    assert len(other) == 1


def test_case_and_suffix_variants_resolve_to_one_club():
    for spelling in ("Coastal union", "COASTAL UNION FC", "coastal  union"):
        assert tanzania.resolve(spelling) == "Coastal Union FC"


def test_an_unknown_club_is_never_guessed_at():
    """A near-match is how 'Le Mans' once became 'Lens'."""
    assert tanzania.resolve("Pamba Jijji United") is None
    page = PAGE + event("7", "2026-09-20 16:00:00", "Wholly New FC",
                        "Simba SC", ["4:00 pm"])
    df = tanzania.parse(page)
    assert tanzania.unknown_clubs(df) == ["Wholly New FC"]


def test_sync_refuses_a_page_with_an_unknown_club(tmp_path):
    page = PAGE + event("7", "2026-09-20 16:00:00", "Wholly New FC",
                        "Simba SC", ["4:00 pm"])
    with pytest.raises(ValueError, match="unrecognised club"):
        tanzania.sync(str(tmp_path), html=page)


# ------------------------------------------------------------------- sync
def test_sync_writes_both_overlays(tmp_path):
    r = tanzania.sync(str(tmp_path), html=PAGE, as_of="2026-09-13")
    res = pd.read_csv(r["results_file"])
    fix = pd.read_csv(r["fixtures_file"])
    assert list(res.columns)[:6] == ["Div", "Date", "HomeTeam", "AwayTeam",
                                     "FTHG", "FTAG"]
    assert len(res) == 2 and len(fix) == 2
    assert set(fix["Time"]) == {"16:00", "19:00"}
    assert set(res["Div"]) == {"TZ1"}


def test_sync_never_shrinks_the_results_already_on_file(tmp_path):
    """The homepage publishes a window, not the season.

    Eight of the ten results on file appeared on it; overwriting rather than
    merging would have silently deleted the other two.
    """
    path = tmp_path / "data" / "manual" / "TZ1.csv"
    path.parent.mkdir(parents=True)
    path.write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG\n"
                    "TZ1,01/09/2026,Singida Black Stars FC,Namungo FC,3,2,,\n",
                    encoding="utf-8")
    r = tanzania.sync(str(tmp_path), html=PAGE, as_of="2026-09-13")
    res = pd.read_csv(r["results_file"])
    assert len(res) == 3
    kept = res[res["HomeTeam"] == "Singida Black Stars FC"].iloc[0]
    assert (int(kept["FTHG"]), int(kept["FTAG"])) == (3, 2)


def test_a_played_pairing_is_not_also_offered_as_a_fixture(tmp_path):
    page = PAGE + event("8", "2026-09-14 16:00:00", "Polisi Tanzania",
                        "Mbeya City", ["4:00 pm"])
    r = tanzania.sync(str(tmp_path), html=page, as_of="2026-09-13")
    fix = pd.read_csv(r["fixtures_file"])
    assert not ((fix["HomeTeam"] == "Polisi Tanzania") &
                (fix["AwayTeam"] == "Mbeya City FC")).any()


def test_sync_drops_fixtures_already_in_the_past(tmp_path):
    r = tanzania.sync(str(tmp_path), html=PAGE, as_of="2026-09-14")
    assert len(pd.read_csv(r["fixtures_file"])) == 0


# --------------------------------------------------------- fixtures overlay
def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Div,Date,Time,HomeTeam,AwayTeam\n" + rows, encoding="utf-8")


def test_overlay_is_merged_on_top_of_the_feed(tmp_path):
    cache = tmp_path / "fixtures.csv"
    _write(cache, "E0,13/09/2026,15:00,Arsenal,Chelsea\n")
    over = tmp_path / "manual" / "fixtures"
    _write(over / "TZ1.csv", "TZ1,13/09/2026,16:00,Pamba Jiji FC,TRA United SC\n")
    fx = fixtures.load_any(None, str(cache), overlay_dir=str(over))
    assert set(fx["Div"]) == {"E0", "TZ1"}


def test_overlay_survives_a_feed_that_does_not_mention_the_league(tmp_path):
    """`refresh` rewrites the cache wholesale; that is what wiped Tanzania."""
    cache = tmp_path / "fixtures.csv"
    over = tmp_path / "manual" / "fixtures"
    _write(over / "TZ1.csv", "TZ1,13/09/2026,16:00,Pamba Jiji FC,TRA United SC\n")
    _write(cache, "E0,13/09/2026,15:00,Arsenal,Chelsea\n")
    fx = fixtures.load_any(None, str(cache), overlay_dir=str(over))
    assert len(fx[fx["Div"] == "TZ1"]) == 1
    # the feed is replaced with one that has never heard of TZ1
    _write(cache, "E0,14/09/2026,15:00,Everton,Fulham\n")
    fx = fixtures.load_any(None, str(cache), overlay_dir=str(over))
    assert len(fx[fx["Div"] == "TZ1"]) == 1


def test_overlay_alone_is_enough_when_there_is_no_feed(tmp_path):
    over = tmp_path / "manual" / "fixtures"
    _write(over / "TZ1.csv", "TZ1,13/09/2026,16:00,Pamba Jiji FC,TRA United SC\n")
    fx = fixtures.load_any(None, str(tmp_path / "nothing.csv"),
                           overlay_dir=str(over))
    assert len(fx) == 1 and fx.iloc[0]["Time"] == "16:00"


def test_a_missing_overlay_directory_is_not_an_error(tmp_path):
    cache = tmp_path / "fixtures.csv"
    _write(cache, "E0,13/09/2026,15:00,Arsenal,Chelsea\n")
    fx = fixtures.load_any(None, str(cache), overlay_dir=str(tmp_path / "nope"))
    assert len(fx) == 1


# ------------------------------------------------------- kick-off display
def test_a_tanzanian_kick_off_is_not_shifted_as_if_it_were_british():
    """16:00 in Dar es Salaam is 16:00 on the card, not 18:00.

    Every other source quotes UK time, so the service converted globally. The
    league's own site publishes EAT already, and a two-hour error on the one
    league the user can attend is the worst kind of wrong.
    """
    from service import app as svc

    row = {"Div": "TZ1", "Date": pd.Timestamp("2026-09-13"), "Time": "16:00"}
    ko, label = svc._local_kickoff(row)
    assert ko.hour == 16 and label.endswith("16:00")

    uk = {"Div": "E0", "Date": pd.Timestamp("2026-09-13"), "Time": "16:00"}
    ko_uk, label_uk = svc._local_kickoff(uk)
    assert ko_uk.hour == 18 and label_uk.endswith("18:00")   # BST -> EAT
