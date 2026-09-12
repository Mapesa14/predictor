"""Tune the shrink knobs for Tanzania, a far more top-heavy league than Europe."""
import pandas as pd

from predictor import backtest
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
XI = 0.0018
print('%-18s %8s %8s %7s %8s %8s' % ('gs / es', '1X2', 'RPS', 'acc', 'O2.5', 'BTTS'))
print('-' * 62)
best, bp = 1e9, None
for gs in (0.25, 0.40, 0.55, 0.70, 1.00):
    for es in (0.70, 0.85, 1.00, 1.10):
        bt = backtest.walk_forward(df, 'TZ1', xi=XI, min_train=200, refit_days=7,
                                   goal_shrink=gs, edge_scale=es,
                                   weights={'goals': 1.0, 'sot': 1.0})
        if not len(bt):
            continue
        s = backtest.score(bt)
        mark = ''
        if s['logloss_1x2'] < best:
            best, bp, mark = s['logloss_1x2'], (gs, es), '  <-'
        print('gs %.2f  es %.2f %8.4f %8.4f %7.3f %8.4f %8.4f%s'
              % (gs, es, s['logloss_1x2'], s['rps'], s['acc'],
                 s['logloss_ou25'], s['logloss_btts'], mark))
print()
print('Europe uses gs 0.40 / es 1.10.  Tanzania prefers gs %.2f / es %.2f'
      % bp)
print('n = %d walk-forward predictions' % s['n'])
