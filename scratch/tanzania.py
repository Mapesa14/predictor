"""Can the engine run a league with results only - no shots, no prices?

That is the Tanzanian case, and most of the world outside Europe's top tier.
Simulated as a 16-team NBC Premier League with a Simba/Yanga duopoly.
"""
import numpy as np
import pandas as pd

from predictor import backtest, markets, model
from predictor.engine import Predictor

TEAMS = ["Simba", "Young Africans", "Azam", "Coastal Union", "Namungo",
         "Singida Black Stars", "KMC", "Tanzania Prisons", "Mtibwa Sugar",
         "Dodoma Jiji", "JKT Tanzania", "Fountain Gate", "Ihefu",
         "Mashujaa", "Kagera Sugar", "Tabora United"]
# a duopoly far sharper than anything in the European data
STRENGTH = {t: (0.85 if i == 0 else 0.80 if i == 1 else 0.35 if i == 2
                else -0.15 - 0.03 * i) for i, t in enumerate(TEAMS)}


def season(start, seed):
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp(start)
    for h in TEAMS:
        for a in TEAMS:
            if h == a:
                continue
            lam = np.exp(0.05 + STRENGTH[h] - 0.6 * STRENGTH[a] + 0.28)
            mu = np.exp(0.05 + STRENGTH[a] - 0.6 * STRENGTH[h])
            fh, fa = rng.poisson(lam), rng.poisson(mu)
            rows.append({"Div": "TZ1", "Date": day, "HomeTeam": h, "AwayTeam": a,
                         "FTHG": int(fh), "FTAG": int(fa),
                         "HTHG": int(rng.binomial(fh, .44)),
                         "HTAG": int(rng.binomial(fa, .44))})
            day += pd.Timedelta(days=1)
    return rows


rows = season("2024-08-15", 1) + season("2025-08-15", 2)
df = pd.DataFrame(rows)
df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
df["Season"] = np.where(df.Date < "2025-07-01", "2024/25", "2025/26")
# deliberately absent: HST/AST, HS/AS, and every odds column
print("columns available:", sorted(df.columns.tolist()))
print("shot columns present:", any(c in df.columns for c in ("HST", "AST")))
print("odds columns present:", any(c.startswith("Avg") for c in df.columns))
print()

p = Predictor.__new__(Predictor)
p.root, p.xi, p.as_of = "", 0.0018, None
p.goal_shrink, p.edge_scale = 0.40, 1.10
p.use_ladder, p._models = False, {}
p.weights = {"goals": 1.0, "sot": 1.0}      # asks for shots that do not exist
p.market_weight = 0.9                        # asks for prices that do not exist
p.df = df

m = p.models("TZ1")["FT"]
print("fitted model type: %s  (fell back to goals only)" % type(m).__name__)
print("home advantage %+.3f   rho %+.3f   matches %d"
      % (m.home_adv, m.rho, m.n_matches))
print()
print("%-22s %7s %8s %8s" % ("Team", "Attack", "Defence", "Rating"))
for t, atk, dfc, rating, pld in model.strength_table(m)[:5]:
    print("%-22s %7.2f %8.2f %8.2f" % (t, atk, dfc, rating))
print("...")
for t, atk, dfc, rating, pld in model.strength_table(m)[-2:]:
    print("%-22s %7.2f %8.2f %8.2f" % (t, atk, dfc, rating))
print()

s = p.predict("Simba", "Young Africans", "TZ1")
print("Simba v Young Africans   %.2f - %.2f   1X2 %.0f/%.0f/%.0f  most likely %s"
      % (s["exp_home"], s["exp_away"], 100 * s["result"]["H"],
         100 * s["result"]["D"], 100 * s["result"]["A"],
         "%d-%d" % s["correct_scores"][0][:2]))
print("  market used: %s   (no prices, so the model stands alone)" % s["market_used"])
s2 = p.predict("Simba", "Tabora United", "TZ1")
print("Simba v Tabora United    %.2f - %.2f   1X2 %.0f/%.0f/%.0f"
      % (s2["exp_home"], s2["exp_away"], 100 * s2["result"]["H"],
         100 * s2["result"]["D"], 100 * s2["result"]["A"]))
print("  every market still present:", len([k for k in s2 if k in
      ("result", "double_chance", "totals", "btts", "asian_handicap",
       "ht_ft", "correct_scores", "winning_margin")]), "of 8")
print()

bt = backtest.walk_forward(df, "TZ1", xi=0.0018, min_train=180, refit_days=7,
                           goal_shrink=0.40, edge_scale=1.10,
                           weights={"goals": 1.0, "sot": 1.0})
sc = backtest.score(bt)
print("walk-forward on the simulated league: n=%d  logloss %.4f  rps %.4f  acc %.3f"
      % (sc["n"], sc["logloss_1x2"], sc["rps"], sc["acc"]))
