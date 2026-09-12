"""Render a fitted prediction as the match card people actually read."""
from __future__ import annotations

from datetime import datetime

from . import leagues

W = 52          # card width
LBL = 22        # label column
TEAM = 18       # team column


def pct(p: float) -> str:
    return "%d%%" % round(100 * p)


def _line(label: str, pick: str, p: float) -> str:
    return "%-*s%-*s%s" % (LBL, label, TEAM, pick, pct(p).rjust(6))


def _rule(ch: str = "-") -> str:
    return ch * W


def _head(title: str) -> list[str]:
    return ["", title]


def card(home: str, away: str, div: str, s: dict, when=None,
         top_scores: int = 5, wide: bool = False) -> str:
    """The standard match card: one screen, the markets people bet most."""
    when = when or datetime.now()
    date = when.strftime("%d %b %Y") if hasattr(when, "strftime") else str(when)
    L = []
    L.append("%s vs %s" % (home.upper(), away.upper()))
    L.append(leagues.name(div))
    L.append(date)
    L.append("")
    L.append("%-*s%-*s%s" % (LBL, "", TEAM, "PREDICTION", "PROBABILITY".rjust(6)))
    L.append(_rule())

    r = s["result"]
    L.append(_line("Home Win", home[:TEAM - 1], r["H"]))
    L.append(_line("Draw", "", r["D"]))
    L.append(_line("Away Win", away[:TEAM - 1], r["A"]))

    dc = s["double_chance"]
    L += _head("Double Chance")
    for k in ("1X", "X2", "12"):
        L.append(_line(k, "", dc[k]))

    t = s["totals"]
    L += _head("Goals")
    for ln in (0.5, 1.5, 2.5, 3.5):
        L.append(_line("Over %.1f" % ln, "", t[ln]["over"]))
    L.append("")
    for ln in (1.5, 2.5, 3.5):
        L.append(_line("Under %.1f" % ln, "", t[ln]["under"]))

    b = s["btts"]
    L += _head("BTTS")
    L.append(_line("Yes", "", b["yes"]))
    L.append(_line("No", "", b["no"]))

    tt = s["team_totals"]
    L += _head("Team Goals")
    for side, team in (("home", home), ("away", away)):
        for ln in (0.5, 1.5):
            L.append(_line("%s Over %.1f" % (team[:LBL - 10], ln), "", tt[side][ln]["over"]))

    if "ht_result" in s:
        h = s["ht_result"]
        L += _head("Half Time")
        L.append(_line("HT Home", "", h["H"]))
        L.append(_line("HT Draw", "", h["D"]))
        L.append(_line("HT Away", "", h["A"]))

    L += _head("Correct Score")
    for i, j, p in s["correct_scores"][:top_scores]:
        L.append(_line("%d-%d" % (i, j), "", p))

    L += _head("EXPECTED GOALS")
    L.append("%s: %.2f" % (home, s["exp_home"]))
    L.append("%s: %.2f" % (away, s["exp_away"]))

    i, j, p = s["correct_scores"][0]
    L += _head("MOST LIKELY SCORE")
    L.append("%d - %d   (%s)" % (i, j, pct(p)))

    if wide:
        L += _extra(home, away, s)
    return "\n".join(L)


def _extra(home: str, away: str, s: dict) -> list[str]:
    """The rest of the book: handicaps, HT/FT, margins, combinations."""
    L = []
    L += _head("Draw No Bet")
    dnb = s["draw_no_bet"]
    L.append("%-*s%s" % (LBL + TEAM, home, pct(dnb["H"]).rjust(6)))
    L.append("%-*s%s" % (LBL + TEAM, away, pct(dnb["A"]).rjust(6)))

    # Both sides are named on one row here, so give each its own column rather
    # than truncating: "Aston Vill 2%" reads like a different club.
    w = max(len(home), len(away))
    L += _head("Asian Handicap (home line)")
    L.append("%-8s%s%s" % ("Line", home.ljust(w + 7), away))
    for ln in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        a = s["asian_handicap"][ln]
        L.append("%-8s%s%s" % ("%+.1f" % ln,
                               pct(a["home"]).rjust(w).ljust(w + 7),
                               pct(a["away"]).rjust(len(away))))

    L += _head("European Handicap")
    for ln in (-1, 1):
        e = s["european_handicap"][ln]
        L.append("%-*s%s" % (LBL, "Home %+d" % ln,
                             "H %s   D %s   A %s" % (pct(e["home"]).rjust(4),
                                                     pct(e["draw"]).rjust(4),
                                                     pct(e["away"]).rjust(4))))

    L += _head("Clean Sheet / Win to Nil")
    cs, wn = s["clean_sheet"], s["win_to_nil"]
    for label, p in (("%s clean sheet" % home, cs["home"]),
                     ("%s clean sheet" % away, cs["away"]),
                     ("%s win to nil" % home, wn["home"]),
                     ("%s win to nil" % away, wn["away"])):
        L.append("%-*s%s" % (LBL + TEAM, label, pct(p).rjust(6)))

    L += _head("Total Goals Odd / Even")
    oe = s["odd_even"]
    L.append(_line("Odd", "", oe["odd"]))
    L.append(_line("Even", "", oe["even"]))

    wm = s["winning_margin"]
    L += _head("Winning Margin")
    for label, k in (("%s by 1" % home, "home_by_1"),
                     ("%s by 2" % home, "home_by_2"),
                     ("%s by 3" % home, "home_by_3"),
                     ("%s by 4+" % home, "home_by_4+"),
                     ("Draw", "draw"),
                     ("%s by 1" % away, "away_by_1"),
                     ("%s by 2" % away, "away_by_2"),
                     ("%s by 3" % away, "away_by_3"),
                     ("%s by 4+" % away, "away_by_4+")):
        L.append("%-*s%s" % (LBL + TEAM, label, pct(wm[k]).rjust(6)))

    # H/D/A is the bookmaker's shorthand; on the card it costs nothing to name
    # the side, and it removes the "which one is A again?" step.
    side = {"H": home, "D": "Draw", "A": away}
    if "ht_ft" in s:
        L += _head("Half Time / Full Time")
        for (h, f), p in sorted(s["ht_ft"].items(), key=lambda kv: -kv[1]):
            L.append("%-*s%s" % (LBL + TEAM, "%s / %s" % (side[h], side[f]),
                                 pct(p).rjust(6)))
        hm = s["half_most_goals"]
        L += _head("Most Goals In")
        L.append(_line("First half", "", hm["first"]))
        L.append(_line("Second half", "", hm["second"]))
        L.append(_line("Equal", "", hm["equal"]))

    L += _head("Result + Goals")
    c = s["combined"]
    tail = {"O2.5": "& Over 2.5", "U2.5": "& Under 2.5", "BTTS": "& BTTS"}
    for k in ("H&O2.5", "H&U2.5", "H&BTTS", "D&O2.5", "D&U2.5", "D&BTTS",
              "A&O2.5", "A&U2.5", "A&BTTS"):
        res, suffix = k.split("&")
        L.append("%-*s%s" % (LBL + TEAM, "%s %s" % (side[res], tail[suffix]),
                             pct(c[k]).rjust(6)))
    return L


def best_bets(s: dict, n: int = 6) -> list:
    """The highest-confidence selection from each market on the card."""
    home = s.get("home", "Home")
    away = s.get("away", "Away")
    cands = []
    r = s["result"]
    k = max(r, key=r.get)
    cands.append(({"H": home + " to win", "D": "Draw",
                   "A": away + " to win"}[k], r[k]))
    dc = s["double_chance"]
    k = max(dc, key=dc.get)
    cands.append(({"1X": home + " or Draw", "X2": "Draw or " + away,
                   "12": "Either team to win"}[k], dc[k]))
    t = s["totals"]
    for ln in (1.5, 2.5, 3.5):
        o, u = t[ln]["over"], t[ln]["under"]
        cands.append(("Over %.1f goals" % ln, o) if o > u
                     else ("Under %.1f goals" % ln, u))
    b = s["btts"]
    cands.append(("Both teams to score", b["yes"]) if b["yes"] > b["no"]
                 else ("Not both teams to score", b["no"]))
    tt = s["team_totals"]
    for side, team in (("home", home), ("away", away)):
        d = tt[side][0.5]
        cands.append(("%s to score" % team, d["over"]) if d["over"] > d["under"]
                     else ("%s to be kept out" % team, d["under"]))
    return sorted(cands, key=lambda kv: -kv[1])[:n]
