"""Grid-search the time-decay parameter xi on out-of-sample log-loss."""
import time
import pandas as pd
from predictor.loader import load
from predictor import backtest, leagues

df = load(r'D:\Downloads July 2026\SoccerData')
divs = leagues.TOP_10
grid = [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0055]
rows = []
for xi in grid:
    t = time.time()
    parts = [backtest.walk_forward(df, d, xi=xi, min_train=180, refit_days=7) for d in divs]
    bt = pd.concat([p for p in parts if len(p)], ignore_index=True)
    s = backtest.score(bt); s['xi'] = xi; s['secs'] = round(time.time() - t, 1)
    rows.append(s)
    print('xi=%.4f n=%d logloss=%.4f rps=%.4f acc=%.3f ou=%.4f btts=%.4f (%.0fs)' % (
        xi, s['n'], s['logloss_1x2'], s['rps'], s['acc'],
        s['logloss_ou25'], s['logloss_btts'], s['secs']), flush=True)
r = pd.DataFrame(rows)
best = r.loc[r['logloss_1x2'].idxmin()]
print('\nBEST xi = %.4f  model logloss %.4f | market %.4f  (model rps %.4f | market %.4f)' % (
    best['xi'], best['logloss_1x2'], best.get('logloss_market', float('nan')),
    best['rps'], best.get('rps_market', float('nan'))))
r.to_csv('scratch/xi_tuning.csv', index=False)
