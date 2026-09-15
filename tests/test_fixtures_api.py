"""Fixtures from API-Football, and cup ties.

The failure that matters here is a prediction for the wrong club, so most of
these pin the name matching: aliases and generic words may map a name; nothing
else may, and two candidates means none.
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import fixtures, record, tips  # noqa: E402
from service import fixtures_api as fa  # noqa: E402
from service import live  # noqa: E402
from tests.test_live import Fake  # noqa: E402
from tests.test_resolve import make_predictor  # noqa: E402


def results(div, teams, date="2026-09-01"):
    return [{"Div": div, "Date": pd.Timestamp(date), "HomeTeam": h, "AwayTeam": a,
             "FTHG": 1, "FTAG": 0}
            for i, h in enumerate(teams) for a in teams[i + 1:]]


@pytest.fixture
def p():
    return make_predictor(
        results("E0", ["Brentford", "Liverpool", "Tottenham", "Man City", "Man United"])
        + results("E2", ["Reading", "Peterboro", "Barnsley", "Boston."])
        + results("EC", ["Aldershot", "Sutton", "Boston", "Halifax"])
        + results("SP1", ["Espanol", "Vallecano", "Real Madrid"])
        + results("SC0", ["Hearts", "Falkirk"])
        + results("TZ1", ["Young Africans SC", "Geita Gold FC"])
        + results("I1", ["Genoa", "Fiorentina", "Pisa"])
        + results("I2", ["Sudtirol", "Bari"])
        + results("EG1", ["Al Ahly SC", "Al Masry Club"], date="2025-05-01"))


def fx(lid, home, away, when="2026-09-15T17:00:00+00:00", status="NS", name="L"):
    return {"fixture": {"date": when, "status": {"short": status}},
            "league": {"id": lid, "name": name, "country": "X"},
            "teams": {"home": {"name": home}, "away": {"name": away}}}


# ------------------------------------------------------------------ names
def test_an_alias_maps_the_provider_spelling(p):
    rows, _ = fa.build_rows([fx(140, "Rayo Vallecano", "Espanyol")], p)
    assert (rows[0]["Div"], rows[0]["HomeTeam"], rows[0]["AwayTeam"]) == \
        ("SP1", "Vallecano", "Espanol")


def test_a_generic_word_may_be_dropped_to_reach_one_club(p):
    rows, _ = fa.build_rows([fx(43, "Aldershot Town", "Sutton Utd")], p)
    assert (rows[0]["HomeTeam"], rows[0]["AwayTeam"]) == ("Aldershot", "Sutton")


def test_two_candidates_means_no_match(p):
    """'Boston Town' reduces to 'boston', which two different clubs fold to."""
    assert fa.resolve_club(p, "Boston Town", "England", ["EC", "E2"]) is None


def test_an_unknown_club_is_reported_and_left_out(p):
    rows, rep = fa.build_rows([fx(39, "Liverpool", "Nowhere Rovers")], p)
    assert rows == []
    assert any("Nowhere Rovers" in u for u in rep["unresolved"])


def test_a_relegated_club_is_found_on_the_ladder(p):
    """Last season's Premier League club now listed in League One."""
    got = fa.resolve_club(p, "Brentford", "England", fa._scope("E2"))
    assert got == ("Brentford", "E0")


# ------------------------------------------------------------- exclusions
def test_a_division_we_do_not_load_is_reported(p):
    rows, rep = fa.build_rows([fx(180, "Stenhousemuir", "Partick")], p)
    assert rows == [] and rep["not_loaded"]


def test_a_league_with_stale_results_is_not_priced(p):
    """Egypt's results in our data stop at May 2025."""
    rows, rep = fa.build_rows([fx(233, "Al Ahly", "AL Masry")], p)
    assert rows == [] and rep["stale"]


def test_postponed_fixtures_are_skipped(p):
    rows, _ = fa.build_rows([fx(39, "Liverpool", "Tottenham", status="PST")], p)
    assert rows == []


# ------------------------------------------------------------------ times
def test_kick_off_is_stored_in_the_divisions_own_zone(p):
    """So leagues.kickoff reads it the same way it reads every other source."""
    tz, _ = fa.build_rows([fx(567, "Young Africans", "Geita Gold",
                              when="2026-09-15T16:00:00+00:00")], p)
    sp, _ = fa.build_rows([fx(140, "Real Madrid", "Espanyol",
                              when="2026-09-15T17:00:00+00:00")], p)
    assert (tz[0]["Date"], tz[0]["Time"]) == ("15/09/2026", "19:00")     # EAT
    assert sp[0]["Time"] == "18:00"                                      # BST


# ------------------------------------------------------------------- cups
def test_a_cup_tie_inside_one_division_is_priced_there(p):
    rows, _ = fa.build_rows([fx(48, "Liverpool", "Tottenham")], p)
    assert (rows[0]["Div"], rows[0]["Comp"], rows[0]["ScaleAlt"]) == \
        ("E0", "League Cup", "")


def test_a_cup_tie_across_the_ladder_keeps_both_scales(p):
    rows, _ = fa.build_rows([fx(48, "Reading", "Brentford")], p)
    assert (rows[0]["Div"], rows[0]["ScaleAlt"]) == ("E0", "E2")


def test_a_cup_tie_with_a_club_we_do_not_carry_is_not_priced(p):
    """Südtirol plays in Serie B, which is not in our data - no record, no price."""
    rows, rep = fa.build_rows([fx(137, "Genoa", "Sudtirol")], p)
    assert rows == []
    assert any("Sudtirol" in u for u in rep["unresolved"])


def test_a_cross_division_tie_with_no_measured_gap_is_not_priced(p, monkeypatch):
    """Today only England has divisions a measured gap joins, so this guard is
    reached by pretending it has none: without a gap, two divisions' ratings
    are on different scales and a price between them would be a guess."""
    monkeypatch.setattr(fa.leagues, "ladder_of", lambda div: None)
    rows, rep = fa.build_rows([fx(48, "Reading", "Brentford")], p)
    assert rows == [] and rep["no_gap"]


def test_cup_ties_stay_out_of_tips_and_the_record(tmp_path):
    cup = {"div": "E0", "league": "League Cup", "comp": "League Cup",
           "date": "2026-09-15T21:00:00+03:00", "home": "Ipswich", "away": "Arsenal",
           "p": {"1": 0.06, "X": 0.15, "2": 0.79},
           "market": {"1": 0.06, "X": 0.15, "2": 0.79}}
    t = tips.build([cup], now=datetime(2026, 9, 15, 6, 0).astimezone())
    assert t["bankers"] == [] and t["long_list"] == []
    r = record.publish([cup], str(tmp_path), as_of="2026-09-15T09:00:00+03:00")
    assert r["written"] == 0 and r["skipped_cup"] == 1


# ------------------------------------------------------------------- sync
@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_API_KEY", "test-key-not-real")
    eng = create_engine("sqlite:///" + str(tmp_path / "live.db").replace("\\", "/"))
    return live.LiveStore(eng)


NOW = datetime(2026, 9, 15, 6, 0)


def test_sync_writes_the_file_and_charges_the_ledger(tmp_path, p, store):
    body = {"errors": [], "response": [fx(140, "Rayo Vallecano", "Espanyol"),
                                       fx(48, "Liverpool", "Tottenham")]}
    fake = Fake(body=body)
    r = fa.sync(str(tmp_path), p, days=1, now=NOW, store=store, transport=fake)
    assert r["fetched"] == ["2026-09-15"] and (r["leagues"], r["cups"]) == (1, 1)
    assert len(pd.read_csv(fa.api_file(str(tmp_path)))) == 2
    assert "date=2026-09-15" in fake.calls[0]
    assert store.spent_since(NOW.replace(hour=0)) == 1


def test_resyncing_the_same_day_does_not_duplicate(tmp_path, p, store):
    body = {"errors": [], "response": [fx(140, "Rayo Vallecano", "Espanyol")]}
    fa.sync(str(tmp_path), p, days=1, now=NOW, store=store, transport=Fake(body=body))
    later = NOW.replace(hour=12)
    fa.sync(str(tmp_path), p, days=1, now=later, store=store, transport=Fake(body=body))
    assert len(pd.read_csv(fa.api_file(str(tmp_path)))) == 1


def test_a_failed_request_never_empties_the_schedule(tmp_path, p, store):
    body = {"errors": [], "response": [fx(140, "Rayo Vallecano", "Espanyol")]}
    fa.sync(str(tmp_path), p, days=1, now=NOW, store=store, transport=Fake(body=body))
    r = fa.sync(str(tmp_path), p, days=1, now=NOW.replace(hour=13), store=store,
                transport=Fake(status=500, body={"errors": {"x": "down"}}))
    assert r["fetched"] == [] and r["errors"]
    assert len(pd.read_csv(fa.api_file(str(tmp_path)))) == 1


def test_sync_sends_nothing_without_a_key(tmp_path, p, store, monkeypatch):
    monkeypatch.delenv("LIVE_API_KEY", raising=False)
    fake = Fake()
    r = fa.sync(str(tmp_path), p, days=1, now=NOW, store=store, transport=fake)
    assert fake.calls == [] and r["written"] is False
    assert not os.path.exists(fa.api_file(str(tmp_path)))


# ---------------------------------------------------------------- priority
def _csv(path, header, *lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def test_published_sources_win_over_api_rows(tmp_path):
    """The feed keeps its closing price; the league site keeps its kick-off."""
    manual = tmp_path / "manual"
    _csv(tmp_path / "fixtures.csv", "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA",
         "E0,15/09/2026,20:00,Liverpool,Tottenham,1.80,3.90,4.20")
    _csv(manual / "fixtures_api.csv", "Div,Date,Time,HomeTeam,AwayTeam,Comp,ScaleAlt",
         "E0,15/09/2026,20:00,Liverpool,Tottenham,,",
         "SP1,15/09/2026,18:00,Vallecano,Espanol,,",
         "TZ1,15/09/2026,21:00,Young Africans SC,Geita Gold FC,,")
    _csv(manual / "fixtures" / "TZ1.csv", "Div,Date,Time,HomeTeam,AwayTeam",
         "TZ1,15/09/2026,19:00,Young Africans SC,Geita Gold FC")
    fx_ = fixtures.load_any(None, str(tmp_path / "fixtures.csv"),
                            overlay_dir=str(manual / "fixtures"))
    e0 = fx_[fx_["Div"] == "E0"]
    assert len(e0) == 1 and float(e0.iloc[0]["AvgH"]) == 1.80
    assert len(fx_[fx_["Div"] == "SP1"]) == 1                # a gap, filled
    tz = fx_[fx_["Div"] == "TZ1"]
    assert len(tz) == 1 and tz.iloc[0]["Time"] == "19:00"


def test_a_shared_town_name_is_not_a_match():
    """Niki Volos and Volos NFC are different clubs; the alias only maps Aris."""
    g = make_predictor(results("G1", ["Volos NFC", "Aris", "Levadeiakos"]))
    assert fa.resolve_club(g, "Niki Volos", "Greece", ["G1"]) is None
    assert fa.resolve_club(g, "Aris Thessalonikis", "Greece", ["G1"]) == ("Aris", "G1")
