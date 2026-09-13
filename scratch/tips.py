"""What would a "clear picks" shortlist actually have delivered?

The product wants a short list of strong calls. Before designing one, measure
what such a list is worth, because there are two very different things it could
mean and only one of them is honest here:

  * high *confidence* - fixtures the model is very sure about. These win often,
    because they are short-priced favourites. There is no edge in them.
  * high *value* - fixtures the model rates better than the price. Already
    measured across this project: the bigger that gap, the more often the model
    is the one that is wrong.

So the rule under test is confidence, plus agreement with the price where a
price exists. Run over walk-forward predictions so nothing has seen its own
result.

    python scratch/tips.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import backtest, loader, market  # noqa: E402

DATA = os.environ.get("SOCCER_DATA", r"D:\Downloads July 2026\SoccerData")
DIVS = ["E0", "SP1", "I1", "D1", "F1", "N1", "P1", "B1", "T1", "G1"]


def build() -> pd.DataFrame:
    df = loader.load(DATA)
    parts = []
    for d in DIVS:
        bt = backtest.walk_forward(df, d, market_weight=0.9)
        if len(bt):
            parts.append(bt)
            print("  %-4s %5d" % (d, len(bt)), flush=True)
    return pd.concat(parts, ignore_index=True)


def annotate(bt: pd.DataFrame) -> pd.DataFrame:
    """Add the pick, whether it won, and what the price said about it."""
    b = bt.dropna(subset=["pH", "pD", "pA"]).copy()
    p = b[["pH", "pD", "pA"]].to_numpy()
    idx = p.argmax(axis=1)
    b["pick"] = pd.Series(idx, index=b.index).map({0: "H", 1: "D", 2: "A"})
    b["conf"] = p.max(axis=1)
    b["won"] = (b["pick"] == b["FTR"]).astype(int)

    # the de-vigged price for the same selection, and the odds you'd be paid
    odds = b[["AvgH", "AvgD", "AvgA"]].apply(pd.to_numeric, errors="coerce")
    ok = odds.notna().all(axis=1)
    b["mkt_conf"] = np.nan
    b["price"] = np.nan
    b["mkt_pick"] = None
    if ok.any():
        o = odds[ok].to_numpy()
        imp = np.array([market.devig(r, "shin") for r in o])
        rows = np.arange(len(imp))
        sel = idx[ok.to_numpy()]
        b.loc[ok, "mkt_conf"] = imp[rows, sel]
        b.loc[ok, "price"] = o[rows, sel]
        b.loc[ok, "mkt_pick"] = pd.Series(
            imp.argmax(axis=1), index=b.index[ok]).map({0: "H", 1: "D", 2: "A"})
    b["agree"] = b["mkt_pick"].isna() | (b["mkt_pick"] == b["pick"])
    return b


def table(b: pd.DataFrame) -> None:
    weeks = max(1.0, (b["Date"].max() - b["Date"].min()).days / 7.0)
    print("\n%d matches over %.0f weeks (%d league-seasons)"
          % (len(b), weeks, b["Div"].nunique()))

    print("\nConfidence threshold, model's favourite:")
    print("%-9s%8s%9s%9s%10s%10s%10s"
          % ("min conf", "picks", "per wk", "hit %", "expected", "flat ROI", "agree %"))
    print("-" * 68)
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        s = b[b["conf"] >= t]
        if len(s) < 30:
            continue
        priced = s.dropna(subset=["price"])
        roi = ((priced["won"] * priced["price"]).sum() / len(priced) - 1) * 100 \
            if len(priced) else float("nan")
        print("%-9.2f%8d%9.1f%8.1f%%%9.1f%%%9.1f%%%9.1f%%"
              % (t, len(s), len(s) / weeks, 100 * s["won"].mean(),
                 100 * s["conf"].mean(), roi, 100 * s["agree"].mean()))

    print("\nSame, but only where the price agrees with the pick:")
    print("%-9s%8s%9s%9s%10s%10s"
          % ("min conf", "picks", "per wk", "hit %", "expected", "flat ROI"))
    print("-" * 58)
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        s = b[(b["conf"] >= t) & b["agree"]]
        if len(s) < 30:
            continue
        priced = s.dropna(subset=["price"])
        roi = ((priced["won"] * priced["price"]).sum() / len(priced) - 1) * 100 \
            if len(priced) else float("nan")
        print("%-9.2f%8d%9.1f%8.1f%%%9.1f%%%9.1f%%"
              % (t, len(s), len(s) / weeks, 100 * s["won"].mean(),
                 100 * s["conf"].mean(), roi))

    print("\nWhere the model and the price disagree on who wins:")
    dis = b[~b["agree"] & b["mkt_pick"].notna()]
    if len(dis):
        priced = dis.dropna(subset=["price"])
        roi = ((priced["won"] * priced["price"]).sum() / len(priced) - 1) * 100
        print("  %d picks, hit %.1f%%, flat ROI %.1f%%"
              % (len(dis), 100 * dis["won"].mean(), roi))

    # The double-chance version: the same shortlist, but "not the underdog".
    print("\nDouble chance (pick or draw) on the same confidence bands:")
    print("%-9s%8s%9s%9s" % ("min conf", "picks", "per wk", "hit %"))
    print("-" * 40)
    for t in (0.45, 0.50, 0.55, 0.60, 0.65):
        s = b[(b["conf"] >= t) & b["agree"] & (b["pick"] != "D")]
        if len(s) < 30:
            continue
        hit = ((s["FTR"] == s["pick"]) | (s["FTR"] == "D")).mean()
        print("%-9.2f%8d%9.1f%8.1f%%" % (t, len(s), len(s) / weeks, 100 * hit))

    print("\nOver/under 2.5, by how far the probability sits from a coin toss:")
    print("%-12s%8s%9s%9s" % ("min prob", "picks", "per wk", "hit %"))
    print("-" * 40)
    for t in (0.55, 0.60, 0.65, 0.70, 0.75):
        over = b[b["pOver25"] >= t]
        under = b[b["pOver25"] <= 1 - t]
        n = len(over) + len(under)
        if n < 30:
            continue
        hit = (over["over25"].sum() + (1 - under["over25"]).sum()) / n
        print("%-12.2f%8d%9.1f%8.1f%%" % (t, n, n / weeks, 100 * hit))


def main() -> int:
    print("walk-forward over %d divisions..." % len(DIVS), flush=True)
    bt = build()
    b = annotate(bt)
    table(b)
    print("\nReminder: hit rate is not profit. A 75%%-confidence favourite is "
          "priced\nnear 1.33, so it has to land about 75%% of the time just to "
          "break even.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
