"""Re-tune the shrink knobs now that blended rates are less noisy than goals alone."""
import pandas as pd

from predictor import backtest, leagues
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
divs = leagues.TOP_10
XI = 0.0018
W = {'goals': 1.0, 'sot': 1.0}


def run(gs, es):
    parts = [backtest.walk_forward(df, d, xi=XI, min_train=180, refit_days=7,
                                   goal_shrink=gs, edge_scale=es, weights=W)
             for d in divs]
    bt = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return backtest.score(bt)


print('--- edge_scale (goal_shrink 0.30) ---', flush=True)
best_es, best = 0.90, 1e9
for es in [0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
    s = run(0.30, es)
    print('es=%.2f  1X2 %.4f  rps %.4f  acc %.3f  O2.5 %.4f  BTTS %.4f'
          % (es, s['logloss_1x2'], s['rps'], s['acc'],
             s['logloss_ou25'], s['logloss_btts']), flush=True)
    if s['logloss_1x2'] < best:
        best_es, best = es, s['logloss_1x2']

print('\n--- goal_shrink (edge_scale %.2f) ---' % best_es, flush=True)
best_gs, bestg = 0.30, 1e9
for gs in [0.20, 0.30, 0.40, 0.50, 0.60]:
    s = run(gs, best_es)
    tot = s['logloss_ou25'] + s['logloss_btts']
    print('gs=%.2f  O2.5 %.4f  BTTS %.4f  sum %.4f  1X2 %.4f'
          % (gs, s['logloss_ou25'], s['logloss_btts'], tot,
             s['logloss_1x2']), flush=True)
    if tot < bestg:
        best_gs, bestg = gs, tot

print('\nCHOSEN goal_shrink=%.2f edge_scale=%.2f' % (best_gs, best_es))
s = run(best_gs, best_es)
print('  1X2 %.4f | market %.4f     rps %.4f | market %.4f     acc %.3f'
      % (s['logloss_1x2'], s['logloss_market'], s['rps'], s['rps_market'], s['acc']))
