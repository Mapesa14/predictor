"""Does a carried-up rating beat the flat promoted prior? Walk forward and see.

For every match where one side is new to its division, predict it three ways:
with the flat prior, with the rating carried from the division it came from,
and with blends in between. Only English divisions can do this - they are the
only ladder where both rungs are loaded.
"""
import numpy as np
import pandas as pd

from predictor import backtest, leagues, markets, model
from predictor.loader import load

df = load(r'D:\Downloads July 2026\SoccerData')
XI, GS, ES = 0.0018, 0.30, 0.90
LADDER = leagues.LADDERS[0]
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]      # 0 = flat prior, 1 = carried rating

rows = []
for div in LADDER:
    dd = df[df.Div == div].sort_values("Date").reset_index(drop=True)
    if len(dd) < 200:
        continue
    fitted, src_models, lastfit = None, None, None
    for _, mt in dd.iloc[180:].iterrows():
        if fitted is None or (mt.Date - lastfit).days >= 14:
            try:
                fitted = model.fit(df, div, "FT", xi=XI, as_of=mt.Date,
                                   goal_shrink=GS, edge_scale=ES)
            except Exception:
                continue
            src_models = {}
            for other in LADDER:
                if other == div or leagues.rating_shift(other, div) is None:
                    continue
                try:
                    src_models[other] = model.fit(df, other, "FT", xi=XI,
                                                  as_of=mt.Date, goal_shrink=GS,
                                                  edge_scale=ES)
                except Exception:
                    pass
            lastfit = mt.Date

        h_new = not fitted.knows(mt.HomeTeam)
        a_new = not fitted.knows(mt.AwayTeam)
        if not (h_new or a_new):
            continue
        # find each newcomer's rating one rung away
        carried = {}
        for t in ([mt.HomeTeam] if h_new else []) + ([mt.AwayTeam] if a_new else []):
            for other in sorted(src_models, key=lambda o: abs(LADDER.index(o) - LADDER.index(div))):
                sm = src_models[other]
                if sm.knows(t) and sm.played.get(t, 0) >= 10:
                    da, dd_ = leagues.rating_shift(other, div)
                    carried[t] = (sm.attack[t] + da,
                                  (sm.defence[t] - sm.mean_defence) + dd_ + fitted.mean_defence)
                    break
        if len(carried) < (h_new + a_new):
            continue        # only score matches where every newcomer is traceable

        rec = {"div": div, "outcome": {"H": 0, "D": 1, "A": 2}[mt.FTR],
               "over25": int(mt.FTHG + mt.FTAG > 2.5)}
        prior = (model.PROMOTED_ATTACK, fitted.mean_defence + model.PROMOTED_DEFENCE)
        for b in BLENDS:
            saved_a, saved_d = dict(fitted.attack), dict(fitted.defence)
            for t, (ca, cd) in carried.items():
                fitted.attack[t] = b * ca + (1 - b) * prior[0]
                fitted.defence[t] = b * cd + (1 - b) * prior[1]
            m = fitted.score_matrix(mt.HomeTeam, mt.AwayTeam)
            r = markets.result(m)
            rec["p%.2f" % b] = [r["H"], r["D"], r["A"]]
            rec["o%.2f" % b] = markets.totals(m, (2.5,))[2.5]["over"]
            fitted.attack, fitted.defence = saved_a, saved_d
        rows.append(rec)

r = pd.DataFrame(rows)
print("matches with a traceable newcomer: %d" % len(r))
if len(r):
    print(r.groupby("div").size().to_string())
    print()
    print("%-22s %8s %8s %8s" % ("blend toward carried", "1X2", "RPS", "O2.5"))
    print("-" * 50)
    for b in BLENDS:
        p = np.array(r["p%.2f" % b].tolist())
        idx = r.outcome.to_numpy()
        ll = -np.mean(np.log(np.clip(p[np.arange(len(p)), idx], 1e-9, 1)))
        rp = np.mean([backtest.rps(p[i], idx[i]) for i in range(len(p))])
        o = np.clip(r["o%.2f" % b].to_numpy(), 1e-9, 1 - 1e-9)
        y = r.over25.to_numpy()
        llo = -np.mean(y * np.log(o) + (1 - y) * np.log(1 - o))
        tag = "  (flat prior)" if b == 0 else ("  (full carry)" if b == 1 else "")
        print("%-22s %8.4f %8.4f %8.4f%s" % ("%.0f%%" % (100 * b), ll, rp, llo, tag))
