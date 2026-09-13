"""The public record. Its only value is that it cannot be quietly rewritten.

Every test here defends one of two claims the product makes to a paying user:
the prediction was written down before kick-off, and nothing has been changed
since. If either can be broken, the record is worth nothing at all.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import leagues, record  # noqa: E402

NOW = "2026-09-13T09:00:00+03:00"


def match(home, away, kickoff, div="E0", p=(0.5, 0.3, 0.2), score="1-0",
          market=None):
    return {"div": div, "league": leagues.name(div), "date": kickoff,
            "home": home, "away": away,
            "p": {"1": p[0], "X": p[1], "2": p[2]},
            "market": market, "o25": 0.52, "btts": 0.48,
            "score": score, "xg": "1.60-1.10"}


def results(rows):
    df = pd.DataFrame(rows, columns=["Div", "Date", "HomeTeam", "AwayTeam",
                                     "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# --------------------------------------------------- written before kick-off
def test_a_fixture_already_under_way_is_refused(tmp_path):
    rows = [match("Arsenal", "Chelsea", "2026-09-13T08:00:00+03:00"),
            match("Leeds", "Everton", "2026-09-13T18:00:00+03:00")]
    r = record.publish(rows, str(tmp_path), as_of=NOW)
    assert r["written"] == 1
    assert r["refused_already_started"] == 1
    assert set(record.load(str(tmp_path))["home"]) == {"Leeds"}


def test_a_fixture_with_no_kick_off_is_refused(tmp_path):
    """Without a time there is nothing for the prediction to predate."""
    rows = [match("Arsenal", "Chelsea", None),
            match("Leeds", "Everton", "2026-09-13T18:00:00+03:00")]
    r = record.publish(rows, str(tmp_path), as_of=NOW)
    assert r["written"] == 1 and r["refused_no_kickoff"] == 1


def test_publishing_twice_adds_nothing(tmp_path):
    rows = [match("Leeds", "Everton", "2026-09-13T18:00:00+03:00")]
    record.publish(rows, str(tmp_path), as_of=NOW)
    again = record.publish(rows, str(tmp_path), as_of=NOW)
    assert again["written"] == 0 and again["already_recorded"] == 1
    assert len(record.load(str(tmp_path))) == 1


def test_a_second_opinion_never_overwrites_the_first(tmp_path):
    """Re-running with different numbers must not move the published ones."""
    ko = "2026-09-13T18:00:00+03:00"
    record.publish([match("Leeds", "Everton", ko, p=(0.5, 0.3, 0.2))],
                   str(tmp_path), as_of=NOW)
    record.publish([match("Leeds", "Everton", ko, p=(0.1, 0.2, 0.7))],
                   str(tmp_path), as_of=NOW)
    df = record.load(str(tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["pH"] == pytest.approx(0.5)


def test_an_east_african_kick_off_is_compared_in_the_right_zone(tmp_path):
    """16:00 EAT is 13:00 UTC: at 14:00 EAT it has not started.

    A naive comparison would have called this one late and refused it, or
    published a match already an hour old - the same class of bug that put a
    Dar es Salaam kick-off on the card two hours out.
    """
    rows = [match("Pamba Jiji FC", "TRA United SC",
                  "2026-09-13T16:00:00+03:00", div="TZ1")]
    r = record.publish(rows, str(tmp_path), as_of="2026-09-13T14:00:00+03:00")
    assert r["written"] == 1
    late = record.publish(
        [match("Kagera Sugar FC", "Geita Gold FC",
               "2026-09-13T16:00:00+03:00", div="TZ1")],
        str(tmp_path), as_of="2026-09-13T17:00:00+03:00")
    assert late["refused_already_started"] == 1


# ------------------------------------------------------------- hash chain
def test_the_chain_is_intact_when_untouched(tmp_path):
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
                    match("Celta", "Malaga", "2026-09-13T19:00:00+03:00",
                          div="SP1")], str(tmp_path), as_of=NOW)
    v = record.verify(str(tmp_path))
    assert v["ok"] and v["rows"] == 2


def test_editing_a_published_probability_breaks_the_chain(tmp_path):
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
                    match("Celta", "Malaga", "2026-09-13T19:00:00+03:00",
                          div="SP1")], str(tmp_path), as_of=NOW)
    p = record.path(str(tmp_path))
    text = open(p, encoding="utf-8").read().replace("0.5", "0.9", 1)
    open(p, "w", encoding="utf-8", newline="").write(text)
    v = record.verify(str(tmp_path))
    assert not v["ok"] and v["broken_at"] == 0


def test_deleting_a_row_breaks_the_chain(tmp_path):
    """Dropping the losers is the easiest way to fake a record."""
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
                    match("Celta", "Malaga", "2026-09-13T19:00:00+03:00",
                          div="SP1")], str(tmp_path), as_of=NOW)
    df = record.load(str(tmp_path))
    df.iloc[1:].to_csv(record.path(str(tmp_path)), index=False)
    assert not record.verify(str(tmp_path))["ok"]


def test_reordering_rows_breaks_the_chain(tmp_path):
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
                    match("Celta", "Malaga", "2026-09-13T19:00:00+03:00",
                          div="SP1")], str(tmp_path), as_of=NOW)
    df = record.load(str(tmp_path))
    df.iloc[::-1].to_csv(record.path(str(tmp_path)), index=False)
    assert not record.verify(str(tmp_path))["ok"]


def test_appending_a_later_day_keeps_the_chain(tmp_path):
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00")],
                   str(tmp_path), as_of=NOW)
    record.publish([match("Inter", "Roma", "2026-09-20T18:00:00+03:00",
                          div="I1")],
                   str(tmp_path), as_of="2026-09-20T09:00:00+03:00")
    v = record.verify(str(tmp_path))
    assert v["ok"] and v["rows"] == 2


# ------------------------------------------------------------- settlement
def test_settlement_joins_results_and_leaves_the_rest_pending(tmp_path):
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
                    match("Celta", "Malaga", "2026-09-13T19:00:00+03:00",
                          div="SP1")], str(tmp_path), as_of=NOW)
    res = results([("E0", "2026-09-13", "Leeds", "Everton", 2, 0)])
    s = record.summary(str(tmp_path), res)
    assert s["published"] == 2 and s["settled"] == 1 and s["pending"] == 1
    assert s["overall"]["n"] == 1


def test_a_correct_favourite_scores_better_than_a_wrong_one(tmp_path):
    """The direction of the metric, pinned: right is a lower log-loss."""
    good = str(tmp_path / "good")
    bad = str(tmp_path / "bad")
    ko = "2026-09-13T18:00:00+03:00"
    record.publish([match("Leeds", "Everton", ko, p=(0.8, 0.15, 0.05))],
                   good, as_of=NOW)
    record.publish([match("Leeds", "Everton", ko, p=(0.05, 0.15, 0.8))],
                   bad, as_of=NOW)
    res = results([("E0", "2026-09-13", "Leeds", "Everton", 2, 0)])   # home win
    assert (record.summary(good, res)["overall"]["logloss_1x2"] <
            record.summary(bad, res)["overall"]["logloss_1x2"])


def test_the_market_is_scored_on_the_same_rows_as_the_model(tmp_path):
    """Scoring the model on everything and the price on its own subset is the
    oldest way to make a model look better than it is."""
    ko = "2026-09-%02dT18:00:00+03:00"
    rows, res = [], []
    for i in range(60):
        day = 13 + i % 5
        mk = {"1": 0.45, "X": 0.28, "2": 0.27} if i % 2 == 0 else None
        rows.append(match("H%d" % i, "A%d" % i, ko % day, market=mk))
        res.append(("E0", "2026-09-%02d" % day, "H%d" % i, "A%d" % i, 1, 0))
    record.publish(rows, str(tmp_path), as_of=NOW)
    s = record.summary(str(tmp_path), results(res))
    assert s["settled"] == 60
    m = s["market"]
    assert m["n_with_price"] == 30
    # the model figure quoted beside the price covers those 15, not all 30
    assert m["logloss_model_same_rows"] != s["overall"]["logloss_1x2"]


def test_an_empty_record_reports_nothing_rather_than_something(tmp_path):
    s = record.summary(str(tmp_path), results([]))
    assert s["published"] == 0 and s["settled"] == 0
    assert s["overall"] == {} and s["recent"] == []
    assert s["chain"]["ok"] is True


def test_a_result_corrected_at_source_flows_through(tmp_path):
    """Outcomes are joined on every read, never stored, so nothing goes stale."""
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00",
                          p=(0.8, 0.15, 0.05))], str(tmp_path), as_of=NOW)
    wrong = record.summary(
        str(tmp_path), results([("E0", "2026-09-13", "Leeds", "Everton", 0, 3)]))
    right = record.summary(
        str(tmp_path), results([("E0", "2026-09-13", "Leeds", "Everton", 3, 0)]))
    assert wrong["recent"][0]["correct"] is False
    assert right["recent"][0]["correct"] is True
    assert record.verify(str(tmp_path))["ok"]      # the record itself unchanged
