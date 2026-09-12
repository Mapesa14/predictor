"""Fit the country bridge and check it beats having no bridge at all."""
import numpy as np
import pandas as pd

from predictor import backtest, euro, leagues, markets, model
from predictor.engine import Predictor

p = Predictor(r'D:\Downloads July 2026\SoccerData')
df = euro.load('data/euro')
cr = df.dropna(subset=['home_div', 'away_div'])
cr = cr[cr.home_div != cr.away_div]
good = euro.resolve(cr, p)
good = good[good.home_ok & good.away_ok]
ds = euro.build_dataset(good, p, min_train=120)
print('calibration ties: %d  (%s -> %s)'
      % (len(ds), ds.Date.min().date(), ds.Date.max().date()))
print()

# ---- out-of-sample: fit on the earlier seasons, score the latest -----------
cut = pd.Timestamp('2025-08-01')
tr, te = ds[ds.Date < cut], ds[ds.Date >= cut]
print('train %d ties before %s | test %d ties after' % (len(tr), cut.date(), len(te)))
fit = euro.fit_offsets(tr)


def score(ds_, off, tag):
    s = off['offsets'] if off else {}
    c = off['intercept'] if off else 0.0
    adv = off['home_adv'] if off else 0.25
    ll, rps_, hit = [], [], []
    for _, x in ds_.iterrows():
        lvl = 0.5 * (x.h_base + x.a_base)
        sh, sa = s.get(x.home_div, 0.0), s.get(x.away_div, 0.0)
        lam = np.exp(c + lvl + x.h_att + x.a_def + sh - sa + adv)
        mu = np.exp(c + lvl + x.a_att + x.h_def + sa - sh)
        m = model.score_matrix_from_rates(lam, mu, -0.03)
        r = markets.result(m)
        pr = np.array([r['H'], r['D'], r['A']])
        o = 0 if x.FTHG > x.FTAG else (1 if x.FTHG == x.FTAG else 2)
        ll.append(-np.log(max(pr[o], 1e-9)))
        rps_.append(backtest.rps(pr, o))
        hit.append(int(pr.argmax() == o))
    print('%-24s logloss %.4f   rps %.4f   acc %.3f'
          % (tag, np.mean(ll), np.mean(rps_), np.mean(hit)))
    return np.mean(ll)


print()
a = score(te, None, 'no bridge (naive)')
b = score(te, fit, 'with country bridge')
print()
print('improvement: %.4f log-loss (%.1f%%)' % (a - b, 100 * (a - b) / a))

# ---- final offsets on everything ------------------------------------------
final = euro.fit_offsets(ds)
# The tuned constants predict_cross reads back from the file; keep them here
# so a re-fit cannot silently drop them (once bitten by exactly that).
final['goal_shrink'] = 1.0      # no level shrinkage: mismatches really run up scores
final['edge_scale'] = 1.1
final['mean_level'] = float(np.log((ds.FTHG.sum() + ds.FTAG.sum()) / (2 * len(ds))))
print()
print('COUNTRY OFFSETS (fitted on all %d ties, zero-sum)' % final['n'])
print('%-24s %8s   %s' % ('League', 'offset', 'goals multiplier vs average'))
print('-' * 66)
for d, v in sorted(final['offsets'].items(), key=lambda kv: -kv[1]):
    print('%-24s %+8.3f   scores %.2fx, concedes %.2fx'
          % (leagues.name(d), v, np.exp(v), np.exp(-v)))
print()
print('European home advantage %+.3f (domestic is around +0.15 to +0.25)'
      % final['home_adv'])
import json
json.dump(final, open('predictor/bridge.json', 'w'), indent=2)
print('wrote predictor/bridge.json')
