"""Does rest / fixture congestion move goals in this data? Measure before building."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from predictor import leagues
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
df = df[df.Div.isin(leagues.TOP_10)].sort_values(['Div', 'Date']).reset_index(drop=True)

# days since each side's previous match in the same division
last = {}
rows = []
for _, m in df.iterrows():
    k_h, k_a = (m.Div, m.HomeTeam), (m.Div, m.AwayTeam)
    rh = (m.Date - last[k_h]).days if k_h in last else np.nan
    ra = (m.Date - last[k_a]).days if k_a in last else np.nan
    rows.append((rh, ra))
    last[k_h] = last[k_a] = m.Date
df['rest_h'], df['rest_a'] = zip(*rows)
d = df.dropna(subset=['rest_h', 'rest_a'])
d = d[(d.rest_h <= 30) & (d.rest_a <= 30)]      # drop winter/summer breaks
print('matches with a usable rest gap: %d' % len(d))
print()

print('goals by home-side rest (days):')
for lo, hi, lab in [(0, 3, '<=3'), (4, 4, '4'), (5, 5, '5'), (6, 7, '6-7'),
                    (8, 10, '8-10'), (11, 30, '11+')]:
    s = d[(d.rest_h >= lo) & (d.rest_h <= hi)]
    if len(s) < 50:
        continue
    print('  %-5s n=%-5d home goals %.3f  away goals %.3f  home win %.3f'
          % (lab, len(s), s.FTHG.mean(), s.FTAG.mean(), (s.FTR == 'H').mean()))
print()

d = d.copy()
d['short_h'] = (d.rest_h <= 3).astype(int)
d['short_a'] = (d.rest_a <= 3).astype(int)
d['diff'] = d.rest_h - d.rest_a
print('short rest (<=3 days) rates: home %.3f  away %.3f'
      % (d.short_h.mean(), d.short_a.mean()))
print()

# Poisson regressions, controlling for nothing but the rest terms
for tgt, own, opp in (('FTHG', 'short_h', 'short_a'), ('FTAG', 'short_a', 'short_h')):
    X = sm.add_constant(d[[own, opp, 'diff']].astype(float))
    r = sm.GLM(d[tgt], X, family=sm.families.Poisson()).fit()
    print('%s ~ own short rest + opponent short rest + rest difference' % tgt)
    for name in [own, opp, 'diff']:
        c, se, p = r.params[name], r.bse[name], r.pvalues[name]
        star = '  <-- significant' if p < 0.05 else ''
        print('   %-9s coef %+.4f  se %.4f  p=%.3f  (x%.3f)%s'
              % (name, c, se, p, np.exp(c), star))
    print()
