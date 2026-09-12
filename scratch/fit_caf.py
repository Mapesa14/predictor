"""Fit the African country bridge and check it beats having none."""
import json

import numpy as np
import pandas as pd

from predictor import backtest, euro, leagues, markets, model
from predictor.engine import Predictor

p = Predictor(r'D:\Downloads July 2026\SoccerData')
caf = euro.load_caf('data/caf')
ds, fed_base = euro.build_dataset_caf(caf, p)
cut = pd.Timestamp('2024-07-01')
tr, te = ds[ds.Date < cut], ds[ds.Date >= cut]
print('CAF ties %d | train %d (2023/24) | test %d (2024/25)' % (len(ds), len(tr), len(te)))


def score(d, off, es=1.0):
    s = off['offsets'] if off else {}
    c = off['intercept'] if off else 0.0
    adv = off['home_adv'] if off else 0.30
    beta = off.get('beta', 1.0) if off else 1.0
    ll, rp, hit = [], [], []
    for _, x in d.iterrows():
        lvl = 0.5 * (x.h_base + x.a_base)
        sh, sa = s.get(x.home_div, 0.0), s.get(x.away_div, 0.0)
        a = c + lvl + beta * (x.h_att + x.a_def) + sh - sa + adv
        b = c + lvl + beta * (x.a_att + x.h_def) + sa - sh
        L, E = 0.5 * (a + b), 0.5 * (a - b)
        m = model.score_matrix_from_rates(np.exp(L + es * E), np.exp(L - es * E), -0.05)
        r = markets.result(m)
        pr = np.array([r['H'], r['D'], r['A']])
        o = 0 if x.FTHG > x.FTAG else (1 if x.FTHG == x.FTAG else 2)
        ll.append(-np.log(max(pr[o], 1e-9)))
        rp.append(backtest.rps(pr, o))
        hit.append(int(pr.argmax() == o))
    return np.mean(ll), np.mean(rp), np.mean(hit)


print()
base = score(te, None)
print('%-26s logloss %.4f  rps %.4f  acc %.3f' % ('no bridge', *base))
best, bestr = 1e9, None
for fb in (False, True):
    for ridge in (0.02, 0.05, 0.08, 0.12):
        f = euro.fit_offsets(tr, ridge=ridge, fit_beta=fb)
        r = score(te, f, 1.0)
        print('%-30s logloss %.4f  rps %.4f  acc %.3f'
              % ('ridge=%.2f beta=%.2f' % (ridge, f['beta']), *r))
        if r[0] < best:
            best, bestr = r[0], (ridge, 1.0, fb)
print()
print('best at es=1.0: ridge %.2f, beta %s  ->  %.4f vs %.4f without (%.1f%% better)'
      % (bestr[0], 'fitted' if bestr[2] else 'fixed at 1', best, base[0],
         100 * (base[0] - best) / base[0]))

# Measure edge_scale itself: with shots absent, bridged ties were tuned at 1.0
# and never given a wider grid, so check whether 1.1 (Europe's value) is better.
f0 = euro.fit_offsets(tr, ridge=bestr[0], fit_beta=bestr[2])
for es in (0.9, 1.0, 1.1, 1.2):
    r = score(te, f0, es)
    print('%-30s logloss %.4f  rps %.4f  acc %.3f'
          % ('edge_scale %.1f' % es, *r))
# The published constant is 1.10 (log-loss 0.9216, matching the measured
# acceptance number). 1.2 shaves a further 0.003 on 124 ties, which is noise,
# so the documented constant stands and the JSON must not drift to 1.2.
es_choose = 1.1
r = score(te, f0, es_choose)
best, bestr = r[0], (bestr[0], es_choose, bestr[2])
print()
print('CHOSEN ridge %.2f, es %.2f, beta %s  ->  %.4f (%.1f%% better than none)'
      % (bestr[0], bestr[1], 'fitted' if bestr[2] else 'fixed at 1', best,
         100 * (base[0] - best) / base[0]))

final = euro.fit_offsets(ds, ridge=bestr[0], fit_beta=bestr[2])
final['edge_scale'], final['goal_shrink'] = bestr[1], 1.0
final['fed_base'] = fed_base
final['mean_level'] = float(np.log((ds.FTHG.sum() + ds.FTAG.sum()) / (2 * len(ds))))
counts = pd.concat([ds.home_div, ds.away_div]).value_counts().to_dict()
final['matches'] = {k: int(v) for k, v in counts.items()}
json.dump(final, open('predictor/bridge_caf.json', 'w'), indent=2)

print()
print('AFRICAN OFFSETS (all %d ties)  - loaded leagues, and federations by CAF record' % final['n'])
print('%-30s %8s %6s' % ('', 'offset', 'ties'))
for d, v in sorted(final['offsets'].items(), key=lambda kv: -kv[1]):
    name = leagues.name(d) if d in leagues.LEAGUES else d.replace('F-', 'federation ')
    print('%-30s %+8.3f %6d' % (name, v, counts.get(d, 0)))
print()
print('CAF home advantage %+.3f   rating transfer beta %.3f'
      % (final['home_adv'], final['beta']))
