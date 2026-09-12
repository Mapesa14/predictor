"""Calibration on the ties that look like Simba/Yanga v a small federation."""
import numpy as np
import pandas as pd

from predictor import backtest, euro, markets, model
from predictor.engine import Predictor

p = Predictor(r'D:\Downloads July 2026\SoccerData')
caf = euro.load_caf('data/caf')
ds, _ = euro.build_dataset_caf(caf, p)
isfed = lambda s: s.str.startswith('F-')
mixed = ds[isfed(ds.home_div) ^ isfed(ds.away_div)].copy()   # league v federation
print('league-v-federation ties: %d' % len(mixed))

# leave-one-season-out both ways, so every mixed tie is scored out of sample
seasons = [(ds.Date < '2024-07-01'), (ds.Date >= '2024-07-01')]
rows = []
for beta_mode in ('fixed', 'fitted'):
    for es in (0.6, 0.8, 1.0):
        ll, rp, exp_goals, act_goals, top = [], [], [], [], []
        for tr_mask in seasons:
            tr = ds[tr_mask]
            te = mixed[~mixed.index.isin(tr.index)]
            f = euro.fit_offsets(tr, ridge=0.05, fit_beta=(beta_mode == 'fitted'))
            s, c, adv, b = f['offsets'], f['intercept'], f['home_adv'], f['beta']
            for _, x in te.iterrows():
                lvl = 0.5 * (x.h_base + x.a_base)
                sh, sa = s.get(x.home_div, 0.), s.get(x.away_div, 0.)
                A = c + lvl + b * (x.h_att + x.a_def) + sh - sa + adv
                B = c + lvl + b * (x.a_att + x.h_def) + sa - sh
                L, E = .5 * (A + B), .5 * (A - B)
                lam, mu = np.exp(L + es * E), np.exp(L - es * E)
                m = model.score_matrix_from_rates(lam, mu, -0.05)
                r = markets.result(m)
                pr = np.array([r['H'], r['D'], r['A']])
                o = 0 if x.FTHG > x.FTAG else (1 if x.FTHG == x.FTAG else 2)
                ll.append(-np.log(max(pr[o], 1e-9)))
                rp.append(backtest.rps(pr, o))
                fav = max(lam, mu)
                exp_goals.append(fav)
                act_goals.append(x.FTHG if lam >= mu else x.FTAG)
                top.append(pr.max())
        rows.append((beta_mode, es, np.mean(ll), np.mean(rp), np.mean(exp_goals),
                     np.mean(act_goals), np.mean(top), len(ll)))

print()
print('%-7s %5s %9s %8s %12s %12s %10s' % ('beta', 'edge', 'logloss', 'rps',
      'fav xG', 'fav scored', 'mean top p'))
for r in rows:
    print('%-7s %5.1f %9.4f %8.4f %12.2f %12.2f %9.0f%%' % (*r[:6], 100 * r[6]))
print('(n = %d scored ties in each row)' % rows[0][7])
