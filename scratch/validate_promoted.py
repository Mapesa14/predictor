"""Does the promoted-team prior beat treating a newcomer as an average side?

Walk forward over the top divisions. Whenever a team appears that the model has
never seen in that division, predict the match twice: once with the measured
prior, once with the neutral assumption, and compare out-of-sample log-loss.
"""
import numpy as np
import pandas as pd

from predictor import leagues, markets, model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
XI, GS, ES = 0.0018, 0.30, 0.90
rows = []

for d in leagues.TOP_10:
    dd = df[df.Div == d].sort_values('Date').reset_index(drop=True)
    fitted, last = None, None
    for _, mt in dd.iloc[180:].iterrows():
        if fitted is None or (mt.Date - last).days >= 7:
            try:
                fitted = model.fit(df, d, 'FT', xi=XI, as_of=mt.Date,
                                   goal_shrink=GS, edge_scale=ES)
            except Exception:
                continue
            last = mt.Date
        h_new = not fitted.knows(mt.HomeTeam)
        a_new = not fitted.knows(mt.AwayTeam)
        if not (h_new or a_new):
            continue
        outcome = {'H': 0, 'D': 1, 'A': 2}[mt.FTR]
        rec = {'div': d, 'outcome': outcome,
               'over25': int(mt.FTHG + mt.FTAG > 2.5)}
        for tag, atk, dfn in (('prior', model.PROMOTED_ATTACK,
                               fitted.mean_defence + model.PROMOTED_DEFENCE),
                              ('neutral', 0.0, fitted.mean_defence)):
            saved = dict(fitted.attack), dict(fitted.defence)
            if h_new:
                fitted.attack[mt.HomeTeam] = atk
                fitted.defence[mt.HomeTeam] = dfn
            if a_new:
                fitted.attack[mt.AwayTeam] = atk
                fitted.defence[mt.AwayTeam] = dfn
            m = fitted.score_matrix(mt.HomeTeam, mt.AwayTeam)
            r = markets.result(m)
            rec[tag] = [r['H'], r['D'], r['A']]
            rec[tag + '_o25'] = markets.totals(m, (2.5,))[2.5]['over']
            fitted.attack, fitted.defence = saved
        rows.append(rec)

r = pd.DataFrame(rows)
print('matches involving a team new to its division: %d' % len(r))
if len(r):
    for tag in ('prior', 'neutral'):
        p = np.array(r[tag].tolist())
        hit = np.clip(p[np.arange(len(p)), r.outcome.to_numpy()], 1e-9, 1)
        ll = -np.mean(np.log(hit))
        rp = np.mean([__import__('predictor.backtest', fromlist=['rps']).rps(
            p[i], r.outcome.iloc[i]) for i in range(len(p))])
        o = np.clip(r[tag + '_o25'].to_numpy(), 1e-9, 1 - 1e-9)
        y = r.over25.to_numpy()
        llo = -np.mean(y * np.log(o) + (1 - y) * np.log(1 - o))
        print('%-8s 1X2 logloss %.4f  rps %.4f  O2.5 logloss %.4f'
              % (tag, ll, rp, llo))
