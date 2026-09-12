"""Where does the model disagree with the price, and who is right?"""
import numpy as np
import pandas as pd

from predictor import backtest, leagues, market, markets, model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
parts = []
for d in leagues.TOP_10:
    bt = backtest.walk_forward(df, d, xi=0.0018, min_train=180, refit_days=7,
                               goal_shrink=0.40, edge_scale=1.10,
                               weights={'goals': 1.0, 'sot': 1.0},
                               market_weight=0.0)          # pure model
    if len(bt):
        parts.append(bt)
bt = pd.concat(parts, ignore_index=True).dropna(subset=['AvgH', 'AvgD', 'AvgA'])
o = bt[['AvgH', 'AvgD', 'AvgA']].to_numpy(float)
mk = np.array([market.devig(r, 'shin') for r in o])
mp = bt[['pH', 'pD', 'pA']].to_numpy(float)
idx = bt['FTR'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy()
gap = np.abs(mp - mk).max(axis=1)
hit_m = np.clip(mp[np.arange(len(mp)), idx], 1e-9, 1)
hit_k = np.clip(mk[np.arange(len(mk)), idx], 1e-9, 1)

print('n = %d matches with prices' % len(bt))
print()
print('%-16s %6s %10s %10s %10s' % ('disagreement', 'n', 'model LL', 'market LL', 'model - market'))
print('-' * 60)
edges = [(0, .03), (.03, .06), (.06, .10), (.10, .15), (.15, 1)]
for lo, hi in edges:
    m = (gap >= lo) & (gap < hi)
    if m.sum() < 20:
        continue
    a, b = -np.log(hit_m[m]).mean(), -np.log(hit_k[m]).mean()
    print('%-16s %6d %10.4f %10.4f %+10.4f'
          % ('%.0f-%.0f pts' % (100 * lo, 100 * hi), m.sum(), a, b, a - b))

print()
print('How often the bigger-probability side was right, when they disagree:')
big = gap >= 0.06
mpick, kpick = mp.argmax(1), mk.argmax(1)
diff = big & (mpick != kpick)
print('  they name different favourites in %d of %d matches (%.0f%%)'
      % (diff.sum(), big.sum(), 100 * diff.sum() / max(big.sum(), 1)))
print('  model favourite correct  %.1f%%' % (100 * (mpick[diff] == idx[diff]).mean()))
print('  market favourite correct %.1f%%' % (100 * (kpick[diff] == idx[diff]).mean()))

print()
print('Disagreement by how much data the model had on the two clubs:')
bt['gap'] = gap
thin = bt.assign(minpld=bt[['Home', 'Away']].notna().sum(axis=1))
for lab, m in [('all', np.ones(len(bt), bool))]:
    pass
print('  mean gap %.3f  |  90th pct %.3f  |  share over 6 pts %.1f%%'
      % (gap.mean(), np.percentile(gap, 90), 100 * (gap >= .06).mean()))
