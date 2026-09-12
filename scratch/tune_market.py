"""Two questions: which de-vig is better, and how much should the price count?"""
import numpy as np
import pandas as pd

from predictor import backtest, leagues, market, markets, model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
d = df[df.Div.isin(leagues.TOP_10)].dropna(subset=['AvgH', 'AvgD', 'AvgA'])
idx = d.FTR.map({'H': 0, 'D': 1, 'A': 2}).to_numpy()
odds = d[['AvgH', 'AvgD', 'AvgA']].to_numpy(float)

print('de-vig methods on %d matches with closing prices:' % len(d))
print('  mean overround %.3f' % np.mean([market.overround(o) for o in odds]))
for name in ('proportional', 'shin'):
    P = np.array([market.devig(o, name) for o in odds])
    ll = -np.mean(np.log(np.clip(P[np.arange(len(P)), idx], 1e-9, 1)))
    rp = np.mean([backtest.rps(P[i], idx[i]) for i in range(len(P))])
    print('  %-13s logloss %.4f  rps %.4f' % (name, ll, rp))

# does reading the price back into goal rates lose anything?
print()
print('round-trip through goal rates (first 400 matches):')
errs, rts = [], []
for _, r in d.head(400).iterrows():
    p = market.devig([r.AvgH, r.AvgD, r.AvgA], 'shin')
    ou = None
    if np.isfinite(r.get('Avg>2.5', np.nan)):
        ou = float(market.devig([r['Avg>2.5'], r['Avg<2.5']], 'shin')[0])
    lam, mu = market.implied_rates(p[0], p[1], p[2], -0.05, ou)
    got = markets.result(model.score_matrix_from_rates(lam, mu, -0.05))
    errs.append(max(abs(got['H'] - p[0]), abs(got['D'] - p[1]), abs(got['A'] - p[2])))
    rts.append((lam, mu))
errs = np.array(errs)
print('  max abs error: median %.5f  p95 %.5f  worst %.5f'
      % (np.median(errs), np.percentile(errs, 95), errs.max()))
print('  implied rates: home %.2f  away %.2f (mean)'
      % (np.mean([x[0] for x in rts]), np.mean([x[1] for x in rts])))
