import numpy as np
import pandas as pd
import os

from predictor import backtest, engine, euro, leagues, loader, markets, model

ROOT = r"D:\Downloads July 2026\SoccerData"
REPO = r"C:\Users\Brian\IdeaProjects\football-predictor"

df = loader.load(ROOT)
print("loaded divs:", sorted(set(df["Div"])))

# ---- walk-forward ----------------------------------------------------------
rows = backtest.prior_scan(df)
print("walk-forward matches:", len(rows))
s = backtest.score_blends(rows)
print(s.set_index("w").to_string())

# ---- UEFA bridge -----------------------------------------------------------
p = engine.Predictor(ROOT)
ud = euro.load(os.path.join(REPO, "data", "euro"))
cr = ud.dropna(subset=["home_div", "away_div"])
cr = cr[cr.home_div != cr.away_div]
good = euro.resolve(cr, p)
good = good[good.home_ok & good.away_ok]
ds = euro.build_dataset(good, p, min_train=120)
print("UEFA dataset:", len(ds))
cut = pd.Timestamp("2025-08-01")
tr, te = ds[ds.Date < cut], ds[ds.Date >= cut]
print("UEFA train/test:", len(tr), len(te))
fit = euro.fit_offsets(tr)


def uefascore(d_, off):
    s = off["offsets"] if off else {}
    c = off["intercept"] if off else 0.0
    adv = off["home_adv"] if off else 0.25
    ll, hit = [], []
    for _, x in d_.iterrows():
        lvl = 0.5 * (x.h_base + x.a_base)
        sh, sa = s.get(x.home_div, 0.0), s.get(x.away_div, 0.0)
        lam = np.exp(c + lvl + x.h_att + x.a_def + sh - sa + adv)
        mu = np.exp(c + lvl + x.a_att + x.h_def + sa - sh)
        m = model.score_matrix_from_rates(lam, mu, -0.03)
        pr = markets.result(m)
        p_ = np.array([pr["H"], pr["D"], pr["A"]])
        o = 0 if x.FTHG > x.FTAG else (1 if x.FTHG == x.FTAG else 2)
        ll.append(-np.log(max(p_[o], 1e-9)))
        hit.append(int(p_.argmax() == o))
    return float(np.mean(ll)), float(np.mean(hit))


print("UEFA base:", uefascore(te, None), "fitted:", uefascore(te, fit))