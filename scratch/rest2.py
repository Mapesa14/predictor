"""Rest, tested against the model's residuals so team strength is controlled for.

The raw correlation says short rest means more goals, but congested fixture
lists belong to the big clubs. The question is whether rest still says anything
once the model's own view of both teams is subtracted.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from predictor import leagues, model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
df = df[df.Div.isin(leagues.TOP_10)].sort_values(['Div', 'Date']).reset_index(drop=True)

last = {}
rh_l, ra_l = [], []
for _, m in df.iterrows():
    kh, ka = (m.Div, m.HomeTeam), (m.Div, m.AwayTeam)
    rh_l.append((m.Date - last[kh]).days if kh in last else np.nan)
    ra_l.append((m.Date - last[ka]).days if ka in last else np.nan)
    last[kh] = last[ka] = m.Date
df['rest_h'], df['rest_a'] = rh_l, ra_l

XI, GS, ES = 0.0018, 0.30, 0.90
rows = []
for d in leagues.TOP_10:
    dd = df[df.Div == d].reset_index(drop=True)
    fitted, lastfit = None, None
    for _, mt in dd.iloc[180:].iterrows():
        if fitted is None or (mt.Date - lastfit).days >= 7:
            try:
                fitted = model.fit(df, d, 'FT', xi=XI, as_of=mt.Date,
                                   goal_shrink=GS, edge_scale=ES)
            except Exception:
                continue
            lastfit = mt.Date
        if not (fitted.knows(mt.HomeTeam) and fitted.knows(mt.AwayTeam)):
            continue
        lam, mu = fitted.rates(mt.HomeTeam, mt.AwayTeam)
        rows.append({'exp_h': lam, 'exp_a': mu, 'gh': mt.FTHG, 'ga': mt.FTAG,
                     'rest_h': mt.rest_h, 'rest_a': mt.rest_a})

r = pd.DataFrame(rows).dropna(subset=['rest_h', 'rest_a'])
r = r[(r.rest_h <= 30) & (r.rest_a <= 30)]
r['short_h'] = (r.rest_h <= 3).astype(float)
r['short_a'] = (r.rest_a <= 3).astype(float)
r['diff'] = (r.rest_h - r.rest_a).astype(float)
print('walk-forward matches with rest data: %d' % len(r))
print('short-rest share: home %.3f away %.3f' % (r.short_h.mean(), r.short_a.mean()))
print()

# Poisson regression with the model's own log-rate as a fixed offset: any
# coefficient here is signal the model does not already have.
for side, tgt, own, opp, off in (('HOME', 'gh', 'short_h', 'short_a', 'exp_h'),
                                 ('AWAY', 'ga', 'short_a', 'short_h', 'exp_a')):
    X = sm.add_constant(r[[own, opp, 'diff']])
    g = sm.GLM(r[tgt], X, family=sm.families.Poisson(),
               offset=np.log(r[off])).fit()
    print('%s goals, model log-rate held as offset:' % side)
    for name in [own, opp, 'diff']:
        c, se, p = g.params[name], g.bse[name], g.pvalues[name]
        mark = '  <-- significant' if p < 0.05 else ''
        print('   %-9s coef %+.4f  se %.4f  p=%.3f  (x%.3f)%s'
              % (name, c, se, p, np.exp(c), mark))
    print()
