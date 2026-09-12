"""How much should the closing price count in the published forecast?

One walk-forward pass stores the model's rates and the price's rates for every
match; blends are then evaluated cheaply at many weights. This is also the
reproduction of the measured acceptance constants: model-only 0.9739, default
0.9568, price-only 0.9563 over 5,948 top-10 matches. The evaluation logic
lives in `predictor.backtest` and is pinned by tests/test_acceptance.py.
"""
import time

from predictor import backtest
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')

t0 = time.time()
r = backtest.prior_scan(df)
print('matches with model and price: %d  (%.0fs)' % (len(r), time.time() - t0))
print()

s = backtest.score_blends(r)
print('%-8s %8s %8s %7s %8s %8s' % ('market w', '1X2', 'RPS', 'acc', 'O2.5', 'BTTS'))
print('-' * 52)
for _, row in s.iterrows():
    tag = '   <- model only' if row.w == 0 else ('   <- price only' if row.w == 1 else '')
    print('%-8.1f %8.4f %8.4f %7.3f %8.4f %8.4f%s'
          % (row.w, row.ll1x2, row.rps, row.acc, row.o25, row.btts, tag),
          flush=True)
print()
best = s.loc[s.ll1x2.idxmin()]
print('BEST market weight %.1f  ->  1X2 log-loss %.4f' % (best.w, best.ll1x2))
print()
print('default is 0.9 -> %.4f' % s.loc[s.w == 0.9, 'll1x2'].iloc[0])