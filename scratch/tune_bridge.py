"""Tune level shrinkage for bridged ties on held-out European matches."""
import json

import numpy as np
import pandas as pd

from predictor import backtest, euro, markets, model
from predictor.engine import Predictor

p = Predictor(r'D:\Downloads July 2026\SoccerData')
df = euro.load('data/euro')
cr = df.dropna(subset=['home_div', 'away_div'])
cr = cr[cr.home_div != cr.away_div]
good = euro.resolve(cr, p)
good = good[good.home_ok & good.away_ok]
ds = euro.build_dataset(good, p, min_train=120)
cut = pd.Timestamp('2025-08-01')
tr, te = ds[ds.Date < cut], ds[ds.Date >= cut]
fit = euro.fit_offsets(tr)
mean_lvl = float(np.log(
    (ds.FTHG.sum() + ds.FTAG.sum()) / (2 * len(ds))))
print('mean European log goal rate: %.3f' % mean_lvl)
print('train %d | test %d' % (len(tr), len(te)))
print()


def score(ds_, off, gs, es, rho=-0.03):
    s, c, adv = off['offsets'], off['intercept'], off['home_adv']
    ll, rp, hit, tot = [], [], [], []
    for _, x in ds_.iterrows():
        lvl0 = 0.5 * (x.h_base + x.a_base)
        sh, sa = s.get(x.home_div, 0.), s.get(x.away_div, 0.)
        loglam = c + lvl0 + x.h_att + x.a_def + sh - sa + adv
        logmu = c + lvl0 + x.a_att + x.h_def + sa - sh
        lvl = 0.5 * (loglam + logmu)
        edge = 0.5 * (loglam - logmu)
        lvl = mean_lvl + gs * (lvl - mean_lvl)
        lam, mu = np.exp(lvl + es * edge), np.exp(lvl - es * edge)
        m = model.score_matrix_from_rates(lam, mu, rho)
        r = markets.result(m)
        pr = np.array([r['H'], r['D'], r['A']])
        o = 0 if x.FTHG > x.FTAG else (1 if x.FTHG == x.FTAG else 2)
        ll.append(-np.log(max(pr[o], 1e-9)))
        rp.append(backtest.rps(pr, o))
        hit.append(int(pr.argmax() == o))
        ov = markets.totals(m, (2.5,))[2.5]['over']
        y = int(x.FTHG + x.FTAG > 2.5)
        tot.append(-(y * np.log(max(ov, 1e-9)) + (1 - y) * np.log(max(1 - ov, 1e-9))))
    return np.mean(ll), np.mean(rp), np.mean(hit), np.mean(tot)


print('%-18s %8s %8s %7s %8s' % ('shrink / edge', '1X2', 'RPS', 'acc', 'O2.5'))
print('-' * 54)
best, bp = 1e9, None
for gs in [0.4, 0.6, 0.8, 1.0]:
    for es in [0.9, 1.0, 1.1]:
        a, b, c_, d = score(te, fit, gs, es)
        print('gs %.1f  es %.1f     %8.4f %8.4f %7.3f %8.4f' % (gs, es, a, b, c_, d))
        if a + d < best:
            best, bp = a + d, (gs, es)
print()
print('BEST goal_shrink=%.1f edge_scale=%.1f' % bp)
final = euro.fit_offsets(ds)
final['goal_shrink'], final['edge_scale'] = bp
final['mean_level'] = mean_lvl
json.dump(final, open('predictor/bridge.json', 'w'), indent=2)
print('wrote predictor/bridge.json')
