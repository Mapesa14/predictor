"""What is each pick category actually worth?

The 1X2 lists were measured before they were offered (scratch/tips.py). Adding
"over 1.5", "GG" and the rest without the same treatment would be taking the
model's word for them, which is the one thing this product does not do.

For every selection the Tips screen could offer, over walk-forward predictions
that have never seen their own result:

  * the hit rate at each confidence threshold, and how many picks a threshold
    leaves per matchday - a rule that fires twice a season is not a product;
  * calibration: the model's own average probability against what happened. A
    category where the model says 80% and 68% come in is not offered at all;
  * flat-stake return where a closing price exists (1X2 and over/under 2.5),
    so no category is sold as free money;
  * the worst single season, because an average hides the year it went wrong.

    python scratch/eval_markets.py            # writes scratch/markets_eval.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import backtest, loader, market  # noqa: E402

DATA = os.environ.get("SOCCER_DATA", r"D:\Downloads July 2026\SoccerData")
DIVS = ["E0", "SP1", "I1", "D1", "F1", "N1", "P1", "B1", "T1", "G1"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markets_eval.json")

# selection -> (probability column, what happened, label, odds column or None)
SELECTIONS = {
    "home":      ("pH", lambda b: (b["FTR"] == "H").astype(int), "Home win", "AvgH"),
    "away":      ("pA", lambda b: (b["FTR"] == "A").astype(int), "Away win", "AvgA"),
    "dc_1x":     ("p1X", lambda b: (b["FTR"] != "A").astype(int), "Home or draw", None),
    "dc_x2":     ("pX2", lambda b: (b["FTR"] != "H").astype(int), "Away or draw", None),
    "dc_12":     ("p12", lambda b: (b["FTR"] != "D").astype(int), "Either team", None),
    "over05":    ("pOver05", lambda b: b["over05"], "Over 0.5 goals", None),
    "over15":    ("pOver15", lambda b: b["over15"], "Over 1.5 goals", None),
    "over25":    ("pOver25", lambda b: b["over25"], "Over 2.5 goals", "AvgO25"),
    "under25":   ("pUnder25", lambda b: 1 - b["over25"], "Under 2.5 goals", "AvgU25"),
    "over35":    ("pOver35", lambda b: b["over35"], "Over 3.5 goals", None),
    "btts_yes":  ("pBTTS", lambda b: b["btts"], "Both teams to score", None),
    "btts_no":   ("pBTTSno", lambda b: 1 - b["btts"], "No both teams to score", None),
}

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def build() -> pd.DataFrame:
    df = loader.load(DATA)
    parts = []
    for d in DIVS:
        bt = backtest.walk_forward(df, d, market_weight=0.9)
        if len(bt):
            parts.append(bt)
            print("  %-4s %5d" % (d, len(bt)), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out["season"] = np.where(out["Date"].dt.month >= 7, out["Date"].dt.year,
                             out["Date"].dt.year - 1)
    return out


def _flat_return(odds: pd.Series, won: pd.Series):
    """1 unit on every pick at the closing price, margin included."""
    ok = odds.notna()
    if not ok.any():
        return None, 0
    o, w = odds[ok].astype(float), won[ok].astype(int)
    return float(np.where(w == 1, o - 1.0, -1.0).sum() / len(o)), int(len(o))


def measure(bt: pd.DataFrame) -> dict:
    matchdays = bt["Date"].nunique()
    out = {"matches": int(len(bt)), "matchdays": int(matchdays),
           "seasons": sorted(int(s) for s in bt["season"].unique()),
           "selections": {}}
    for key, (pcol, outcome, label, ocol) in SELECTIONS.items():
        if pcol not in bt.columns:
            continue
        b = bt.dropna(subset=[pcol]).copy()
        b["won"] = outcome(b).astype(int)
        rows = []
        for t in THRESHOLDS:
            sel = b[b[pcol] >= t]
            if not len(sel):
                continue
            by_season = sel.groupby("season")["won"].mean()
            flat, priced = _flat_return(
                sel[ocol] if ocol and ocol in sel.columns else pd.Series(dtype=float),
                sel["won"])
            rows.append({
                "threshold": t,
                "n": int(len(sel)),
                "per_matchday": round(len(sel) / matchdays, 2),
                "hit": round(float(sel["won"].mean()), 4),
                "model_said": round(float(sel[pcol].mean()), 4),
                "worst_season": round(float(by_season.min()), 4) if len(by_season) else None,
                "seasons": {int(s): round(float(v), 4) for s, v in by_season.items()},
                "flat_return": None if flat is None else round(flat, 4),
                "priced": priced,
            })
        out["selections"][key] = {"label": label, "prob_col": pcol, "rows": rows}
    return out


def main() -> None:
    print("walking forward over %d divisions..." % len(DIVS), flush=True)
    bt = build()
    res = measure(bt)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("\n%d matches, %d matchdays, seasons %s\n"
          % (res["matches"], res["matchdays"], res["seasons"]))
    hdr = "%-10s %-22s %5s %6s %6s %9s %9s %9s"
    print(hdr % ("selection", "label", "min", "n", "/day", "hit", "model", "worst yr"))
    for key, s in res["selections"].items():
        for r in s["rows"]:
            if r["n"] < 100:
                continue
            print(hdr % (key, s["label"][:22], "%.2f" % r["threshold"], r["n"],
                         "%.1f" % r["per_matchday"], "%.1f%%" % (100 * r["hit"]),
                         "%.1f%%" % (100 * r["model_said"]),
                         "%.1f%%" % (100 * r["worst_season"])))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
