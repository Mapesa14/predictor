"""Does the engine hold up in a league dominated by one or two clubs?

Tanzania's top flight is a Simba / Yanga duopoly. Before recommending the model
be pointed at it, measure how accuracy varies with competitive balance across
the divisions already in hand - Scotland (Celtic/Rangers) and Greece
(Olympiakos/PAOK/AEK) are the closest analogues available.
"""
import numpy as np
import pandas as pd

from predictor import backtest, leagues, model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
XI, GS, ES = 0.0018, 0.40, 1.10
W = {'goals': 1.0, 'sot': 1.0}
rows = []

for div in sorted(set(df.Div)):
    d = df[df.Div == div]
    if len(d) < 400:
        continue
    # competitive balance, measured two ways on the latest full season
    season = sorted(set(d.Season))[-2] if len(set(d.Season)) > 1 else sorted(set(d.Season))[-1]
    s = d[d.Season == season]
    pts = {}
    for _, m in s.iterrows():
        pts.setdefault(m.HomeTeam, 0)
        pts.setdefault(m.AwayTeam, 0)
        if m.FTR == 'H':
            pts[m.HomeTeam] += 3
        elif m.FTR == 'A':
            pts[m.AwayTeam] += 3
        else:
            pts[m.HomeTeam] += 1
            pts[m.AwayTeam] += 1
    p = np.array(sorted(pts.values(), reverse=True), dtype=float)
    n = len(p)
    top2_share = p[:2].sum() / p.sum() * n / 2      # 1.0 = perfectly even
    m_all = model.fit(df, div, 'FT', xi=XI, goal_shrink=1.0, edge_scale=1.0)
    spread = float(np.std([m_all.attack[t] - m_all.defence[t] for t in m_all.teams]))

    bt = backtest.walk_forward(df, div, xi=XI, min_train=180, refit_days=7,
                               goal_shrink=GS, edge_scale=ES, weights=W)
    if not len(bt):
        continue
    sc = backtest.score(bt)
    rows.append({'div': div, 'league': leagues.name(div), 'teams': n,
                 'top2': top2_share, 'spread': spread, 'n': sc['n'],
                 'logloss': sc['logloss_1x2'], 'rps': sc['rps'], 'acc': sc['acc'],
                 'mkt': sc.get('logloss_market', np.nan)})

r = pd.DataFrame(rows).sort_values('top2', ascending=False)
r['vs_mkt'] = r.logloss - r.mkt
print('%-22s %5s %6s %7s %6s %8s %8s %8s' %
      ('League', 'Tms', 'Top2', 'Spread', 'Acc', 'LogLoss', 'Market', 'Gap'))
print('-' * 78)
for _, x in r.iterrows():
    print('%-22s %5d %6.2f %7.2f %6.3f %8.4f %8.4f %+8.4f'
          % (x.league, x.teams, x.top2, x.spread, x.acc, x.logloss,
             x.mkt, x.vs_mkt))
print()
print('correlation, top-2 dominance vs ...')
print('  model log-loss   %+.2f   (lower log-loss = easier to predict)' % r.top2.corr(r.logloss))
print('  accuracy         %+.2f' % r.top2.corr(r.acc))
print('  gap to market    %+.2f   (does imbalance hurt us more than the book?)' % r.top2.corr(r.vs_mkt))
print('  rating spread    %+.2f' % r.top2.corr(r.spread))
