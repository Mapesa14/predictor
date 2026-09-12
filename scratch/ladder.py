"""Measure how much weaker a division is than the one above it.

Clubs that move between two divisions are the bridge: fit each division
separately, then compare a mover's rating on both sides of the move. Pool
across every club that made the same move.
"""
import numpy as np
import pandas as pd

from predictor import model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
LADDER = [("E0", "E1"), ("E1", "E2"), ("E2", "E3"), ("E3", "EC")]

per_season = {}
for div in {d for pair in LADDER for d in pair}:
    dd = df[df.Div == div]
    for season in sorted(set(dd.Season)):
        sub = dd[dd.Season == season]
        if len(sub) < 150:
            continue
        try:
            m = model.fit(sub, div, "FT", xi=0.0)
        except Exception:
            continue
        atk = np.array([m.attack[t] for t in m.teams])
        dfn = np.array([m.defence[t] for t in m.teams])
        per_season[(div, season)] = {
            t: (m.attack[t] - atk.mean(), m.defence[t] - dfn.mean()) for t in m.teams}

print('division-seasons fitted: %d' % len(per_season))
print()
print('%-10s %6s   %-22s %-22s' % ('move', 'movers', 'attack shift', 'defence shift'))
print('-' * 66)
offsets = {}
for upper, lower in LADDER:
    rows = []
    seasons = sorted({s for (d, s) in per_season if d in (upper, lower)})
    for prev, cur in zip(seasons, seasons[1:]):
        for a_div, b_div in ((upper, lower), (lower, upper)):
            A = per_season.get((a_div, prev))
            B = per_season.get((b_div, cur))
            if not A or not B:
                continue
            for t in set(A) & set(B):
                # rating in the lower division minus rating in the upper one
                if a_div == upper:      # relegated: upper then lower
                    rows.append((B[t][0] - A[t][0], B[t][1] - A[t][1]))
                else:                   # promoted: lower then upper
                    rows.append((A[t][0] - B[t][0], A[t][1] - B[t][1]))
    if len(rows) < 8:
        print('%-10s %6d   (too few movers to estimate)' % (upper + '/' + lower, len(rows)))
        continue
    r = np.array(rows)
    da, dd_ = r[:, 0].mean(), r[:, 1].mean()
    sa, sd = r[:, 0].std() / np.sqrt(len(r)), r[:, 1].std() / np.sqrt(len(r))
    offsets[(upper, lower)] = (da, dd_)
    print('%-10s %6d   %+.3f (se %.3f)        %+.3f (se %.3f)'
          % (upper + '/' + lower, len(r), da, sa, dd_, sd))

print()
print('Reading: a club rates this much higher on attack and lower on defence in')
print('the lower division, so moving up costs it that much.')
print()
print('OFFSETS =', {('%s' % u, '%s' % l): (round(a, 3), round(b, 3))
                   for (u, l), (a, b) in offsets.items()})
