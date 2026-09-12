"""Extend the goal_shrink grid downward, then tune edge_scale."""
import pandas as pd
from predictor.loader import load
from predictor import backtest, leagues

df = load(r'D:\Downloads July 2026\SoccerData')
divs = leagues.TOP_10
XI = 0.0018


def run(gs, es):
    parts = [backtest.walk_forward(df, d, xi=XI, min_train=180, refit_days=7,
                                   goal_shrink=gs, edge_scale=es) for d in divs]
    bt = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return backtest.score(bt)


print('--- goal_shrink (edge_scale = 1.0) ---', flush=True)
best_gs, best = 1.0, 1e9
for gs in [0.0, 0.15, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
    s = run(gs, 1.0)
    comb = s['logloss_ou25'] + s['logloss_btts']
    print('gs=%.2f  O2.5 %.4f  BTTS %.4f  sum %.4f  1X2 %.4f  rps %.4f'
          % (gs, s['logloss_ou25'], s['logloss_btts'], comb,
             s['logloss_1x2'], s['rps']), flush=True)
    if comb < best:
        best_gs, best = gs, comb

print('\n--- edge_scale (goal_shrink = %.2f) ---' % best_gs, flush=True)
best_es, bll = 1.0, 1e9
for es in [0.80, 0.90, 0.95, 1.00, 1.05, 1.10]:
    s = run(best_gs, es)
    print('es=%.2f  1X2 %.4f  rps %.4f  acc %.3f  O2.5 %.4f  BTTS %.4f'
          % (es, s['logloss_1x2'], s['rps'], s['acc'],
             s['logloss_ou25'], s['logloss_btts']), flush=True)
    if s['logloss_1x2'] < bll:
        best_es, bll = es, s['logloss_1x2']
print('\nCHOSEN goal_shrink=%.2f edge_scale=%.2f' % (best_gs, best_es))
