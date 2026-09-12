"""How much weight should shot data carry against goals? Decide out-of-sample."""
import time

import pandas as pd

from predictor import backtest, leagues
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
divs = leagues.TOP_10
XI, GS, ES = 0.0018, 0.30, 0.90


def run(weights):
    parts = [backtest.walk_forward(df, d, xi=XI, min_train=180, refit_days=7,
                                   goal_shrink=GS, edge_scale=ES, weights=weights)
             for d in divs]
    bt = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return backtest.score(bt), bt


print('%-26s %8s %8s %8s %8s' % ('weights (goals/sot/shots)', '1X2', 'RPS', 'O2.5', 'BTTS'))
print('-' * 64)
grids = [
    {'goals': 1.0},
    {'goals': 1.0, 'sot': 0.5},
    {'goals': 1.0, 'sot': 1.0},
    {'goals': 1.0, 'sot': 1.5},
    {'goals': 1.0, 'sot': 2.0},
    {'goals': 1.0, 'sot': 3.0},
    {'goals': 0.0, 'sot': 1.0},
    {'goals': 1.0, 'sot': 1.0, 'shots': 0.5},
    {'goals': 1.0, 'sot': 1.5, 'shots': 0.5},
    {'goals': 1.0, 'sot': 1.0, 'shots': 1.0},
]
best, bestw = 1e9, None
for wts in grids:
    t = time.time()
    s, _ = run(wts)
    lab = '%.1f/%.1f/%.1f' % (wts.get('goals', 0), wts.get('sot', 0), wts.get('shots', 0))
    print('%-26s %8.4f %8.4f %8.4f %8.4f   (%.0fs)'
          % (lab, s['logloss_1x2'], s['rps'], s['logloss_ou25'],
             s['logloss_btts'], time.time() - t), flush=True)
    tot = s['logloss_1x2'] + s['logloss_ou25'] + s['logloss_btts']
    if tot < best:
        best, bestw = tot, wts
print()
print('BEST combined:', bestw)
s, _ = run(bestw)
print('  1X2 %.4f | market %.4f    RPS %.4f | market %.4f'
      % (s['logloss_1x2'], s.get('logloss_market', float('nan')),
         s['rps'], s.get('rps_market', float('nan'))))
