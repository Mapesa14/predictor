"""Does a finished match read as finished?

A fixture whose kick-off had passed said "started" until the day rolled over,
so a match that ended three hours ago still looked like it was being played.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import matchstate  # noqa: E402

KO = datetime(2026, 10, 9, 18, 0, tzinfo=timezone.utc)


def at(minutes):
    return KO + timedelta(minutes=minutes)


def test_before_kick_off_it_is_upcoming():
    assert matchstate.of(KO, at(-1)) == "upcoming"


def test_during_the_match_it_is_live():
    assert matchstate.of(KO, at(1)) == "live"
    assert matchstate.of(KO, at(100)) == "live"


def test_long_after_kick_off_it_has_ended_even_with_no_score():
    """The symptom that started this: no result had reached us, and the row
    claimed the match was under way for the rest of the day."""
    assert matchstate.of(KO, at(matchstate.FULL_MATCH_MINUTES + 1)) == "ended"


def test_a_final_score_ends_it_whatever_the_clock_says():
    assert matchstate.of(KO, at(50), has_final=True) == "ended"


def test_the_cushion_outlasts_stoppage_and_a_late_kick_off():
    """Calling a match over while it is still being played is the worse error,
    so the window runs well past 90 minutes plus half-time."""
    assert matchstate.FULL_MATCH_MINUTES >= 120
    assert matchstate.of(KO, at(115)) == "live"


def test_a_fixture_with_no_kick_off_is_never_called_live():
    assert matchstate.of(None, at(10)) == "upcoming"
