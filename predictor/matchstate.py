"""Upcoming, in play, or over - and the final score once anything carries it.

A fixture used to be "started" from kick-off until the day rolled over, so a
match that finished three hours ago still sat on the slate looking like it was
under way. Kick-off time alone cannot tell the difference; two things settle it:

  * a final score from any source we hold - the official results the engine
    fits on, or API-Football's finals - which makes it over, whatever the clock
    says;
  * otherwise the clock, generously: 90 minutes plus half-time plus stoppage
    plus a cushion. Past that a match is over even if no score has reached us
    yet, and the screen says "ended" rather than claiming it is still running.

The cushion is deliberately long. Calling a match finished while it is still
being played is a worse error than being late to say it ended: one is wrong on
screen, the other is merely slow.
"""
from __future__ import annotations

from datetime import datetime, timedelta

# 90 minutes + 15 half-time + stoppage + a cushion for a delayed kick-off.
FULL_MATCH_MINUTES = 130


def of(kickoff, now: datetime, has_final: bool = False,
       window: int = FULL_MATCH_MINUTES) -> str:
    """"upcoming", "live" or "ended" for one fixture."""
    if has_final:
        return "ended"
    if kickoff is None or now is None:
        return "upcoming"
    if now < kickoff:
        return "upcoming"
    return "ended" if now >= kickoff + timedelta(minutes=window) else "live"
