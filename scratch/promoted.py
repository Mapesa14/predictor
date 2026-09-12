"""Measure how a newly promoted side performs in its first top-division season.

Only tier-1 divisions count: in a lower tier a "new" team is often one relegated
from above, who is good, and that would poison the estimate.
"""
import numpy as np
import pandas as pd

from predictor import leagues, model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
top = [d for d, m in leagues.LEAGUES.items() if m[2] == 1]
rows = []
for d in top:
    dd = df[df.Div == d]
    seasons = sorted(set(dd.Season))
    for prev, cur in zip(seasons, seasons[1:]):
        before = set(dd[dd.Season == prev].HomeTeam) | set(dd[dd.Season == prev].AwayTeam)
        sub = dd[dd.Season == cur]
        if len(sub) < 100:
            continue
        m = model.fit(sub, d, 'FT', xi=0.0)
        atk = np.array([m.attack[t] for t in m.teams])
        dfn = np.array([m.defence[t] for t in m.teams])
        for t in (set(sub.HomeTeam) | set(sub.AwayTeam)) - before:
            rows.append({'div': d, 'season': cur, 'team': t,
                         'atk_c': m.attack[t] - atk.mean(),
                         'dfn_c': m.defence[t] - dfn.mean()})

r = pd.DataFrame(rows)
n = len(r)
print('promoted team-seasons: %d across %d top divisions' % (n, r['div'].nunique()))
print('attack  vs league mean %+.3f  (se %.3f)' % (r.atk_c.mean(), r.atk_c.std() / np.sqrt(n)))
print('defence vs league mean %+.3f  (se %.3f)' % (r.dfn_c.mean(), r.dfn_c.std() / np.sqrt(n)))
print('scores %.2fx and concedes %.2fx the league average'
      % (np.exp(r.atk_c.mean()), np.exp(r.dfn_c.mean())))
print()
print(r.groupby('div')[['atk_c', 'dfn_c']].agg(['mean', 'count']).round(3).to_string())
