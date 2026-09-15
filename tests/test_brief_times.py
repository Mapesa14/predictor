"""Brief kick-off times and competition labels.

A brief once read every fixture's time as UK time. The Tanzanian league site
publishes East Africa Time, so a 19:00 Dar es Salaam kick-off printed at 21:00
- the same class of bug already fixed in the service, surviving in the CLI.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import cli, fixtures  # noqa: E402

EAT = "Africa/Dar_es_Salaam"


def row(div, time, date="2026-09-15", **extra):
    r = {"Div": div, "Date": pd.Timestamp(date), "Time": time}
    r.update(extra)
    return r


def test_a_tanzanian_kick_off_is_not_shifted_as_if_it_were_british():
    _, hhmm = cli._local_kickoff(row("TZ1", "19:00"), EAT)
    assert hhmm == "19:00"


def test_a_uk_feed_kick_off_converts_to_east_africa_time():
    _, hhmm = cli._local_kickoff(row("SP1", "18:00"), EAT)     # BST -> EAT
    assert hhmm == "20:00"


def test_a_late_uk_kick_off_lands_on_the_next_day_in_east_africa():
    ts, hhmm = cli._local_kickoff(row("E0", "22:30"), EAT)
    assert hhmm == "00:30" and ts.day == 16


def test_a_cup_tie_is_labelled_with_its_competition():
    """Priced on the Premier League's scale, but not a Premier League match."""
    assert cli._comp_label(row("E0", "19:45", Comp="League Cup"), "E0") == \
        "League Cup (England)"
    assert cli._comp_label(row("E0", "19:45"), "E0") == "Premier League (England)"


def test_a_blank_competition_falls_back_to_the_league():
    assert cli._comp_label(row("SP1", "18:00", Comp=float("nan")), "SP1") == \
        "La Liga (Spain)"


def test_the_fixtures_loader_keeps_the_competition_column(tmp_path):
    p = tmp_path / "fixtures.csv"
    p.write_text("Div,Date,Time,HomeTeam,AwayTeam,Comp\n"
                 "E0,15/09/2026,19:45,Liverpool,Tottenham,League Cup\n",
                 encoding="utf-8")
    fx = fixtures.from_csv(str(p))
    assert fx.iloc[0]["Comp"] == "League Cup"
