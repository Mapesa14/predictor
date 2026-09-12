"""Should the market's share of the forecast rise with the disagreement?"""
import numpy as np
import pandas as pd

from predictor import backtest, leagues, market
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
parts = []
for d in leagues.TOP_10:
    b = backtest.walk_forward(df, d, xi=0.0018, min_train=180, refit_days=7,
                              goal_shrink=0.40, edge_scale=1.10,
                              weights={'goals': 1.0, 'sot': 1.0},
                              market_weight=0.0)
    if len(b):
        parts.append(b)
bt = pd.concat(parts, ignore_index=True).dropna(subset=['AvgH', 'AvgD', 'AvgA'])
bt = bt.sort_values('Date').reset_index(drop=True)
mk = np.array([market.devig(r, 'shin')
               for r in bt[['AvgH', 'AvgD', 'AvgA']].to_numpy(float)])
mp = bt[['pH', 'pD', 'pA']].to_numpy(float)
idx = bt['FTR'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy()
gap = np.abs(mp - mk).max(axis=1)
cut = int(len(bt) * 0.5)
tr, te = slice(0, cut), slice(cut, None)


def ll(w, sl):
    w = np.clip(w, 0, 1)
    p = (1 - w)[:, None] * mp[sl] + w[:, None] * mk[sl]
    return -np.log(np.clip(p[np.arange(len(p)), idx[sl]], 1e-9, 1)).mean()


print('n train %d | test %d' % (cut, len(bt) - cut))
print()
print('%-34s %10s %10s' % ('rule', 'train LL', 'test LL'))
print('-' * 56)
best_fixed = min(((ll(np.full(cut, w), tr), w) for w in np.arange(.70, 1.001, .05)))
for w in (0.0, 0.80, 0.90, 1.0):
    print('%-34s %10.4f %10.4f'
          % ('fixed market weight %.2f' % w,
             ll(np.full(cut, w), tr), ll(np.full(len(bt) - cut, w), te)))
print('%-34s %10.4f %10.4f'
      % ('fixed, best on train (%.2f)' % best_fixed[1],
         best_fixed[0], ll(np.full(len(bt) - cut, best_fixed[1]), te)))
print()
grid = [(w0, k) for w0 in (0.5, 0.6, 0.7, 0.8) for k in (1.0, 2.0, 3.0, 4.0)]
scored = sorted((ll(w0 + k * gap[tr], tr), w0, k) for w0, k in grid)
for s, w0, k in scored[:3]:
    print('%-34s %10.4f %10.4f'
          % ('adaptive  w = %.1f + %.0f x gap' % (w0, k), s,
             ll(w0 + k * gap[te], te)))
bs, bw0, bk = scored[0]
print()
fixed_te = ll(np.full(len(bt) - cut, 0.9), te)
ada_te = ll(bw0 + bk * gap[te], te)
print('best adaptive beats fixed 0.90 on the held-out half by %.4f log-loss (%.1f%%)'
      % (fixed_te - ada_te, 100 * (fixed_te - ada_te) / fixed_te))
print('market alone on the same half: %.4f' % ll(np.ones(len(bt) - cut), te))
