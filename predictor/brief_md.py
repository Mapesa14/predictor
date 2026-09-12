"""Render a slate of upcoming fixtures as Markdown.

Same inputs as `brief.build`, so the PDF and the Markdown can never drift
apart: the caller assembles one set of rows and picks a renderer by file
extension.
"""
from __future__ import annotations

from datetime import datetime


def _pct(x) -> str:
    return "%d%%" % round(100 * float(x))


def _esc(s) -> str:
    return str(s).replace("|", r"\|")


def build(rows, out_path: str, window_label: str, generated: datetime,
          meta: dict, dissent=None, skipped=None, promoted=None,
          empty_note: str | None = None, ratings=None, cross=None) -> str:
    """Write the slate to `out_path` as Markdown and return the text."""
    dissent = dissent or []
    skipped = skipped or []
    promoted = promoted or []
    cross = cross or []
    mw = round(100 * meta.get("market_weight", 0))
    L = []

    L.append("# Football predictions")
    L.append("")
    L.append("**%s**" % window_label)
    L.append("")
    L.append("Generated %s &middot; Dixon-Coles bivariate Poisson fitted on "
             "%s historical matches." % (generated.strftime("%a %d %b %Y, %H:%M"),
                                         meta.get("n_matches", "")))
    if mw:
        L.append("Where a closing price exists, **%d%%** of the forecast is "
                 "taken from it and %d%% from the model." % (mw, 100 - mw))
    L.append("")
    if mw:
        L.append("> This is analysis, not tipping. Blended with the price as "
                 "above, the engine essentially tracks the market &mdash; "
                 "0.9568 log-loss against the market's 0.9558 over 5,948 "
                 "walk-forward matches. Unblended it scores 0.9739, and betting "
                 "its disagreements with the price lost 12.2% on flat stakes. "
                 "Percentages mean what they say; they are not an edge.")
    else:
        L.append("> This is analysis, not tipping. **No closing price exists "
                 "for these**, so every number is the model alone. On its own "
                 "the engine scores 0.9739 log-loss over 5,948 walk-forward "
                 "matches against the market's 0.9569 &mdash; well calibrated, "
                 "and a little behind a bookmaker who can see team news. "
                 "Percentages mean what they say; they are not an edge.")
    L.append("")

    if not rows and cross:
        return _write(out_path, L + _cross_section(cross) + _tail(promoted, skipped))
    if not rows:
        L.append("## No fixtures in this window")
        L.append("")
        if empty_note:
            L.append(empty_note)
            L.append("")
        if ratings:
            L.append("---")
            L.append("")
            L.append("## What the engine currently rates")
            L.append("")
            L.append("Nothing to forecast today does not mean nothing is known. "
                     "These are the live ratings every prediction is built "
                     "from, as of this run. **Attack** is goals scored relative "
                     "to the league average, **defence** is goals conceded — so "
                     "lower defence is better — and **rating** is the ratio of "
                     "the two.")
            L.append("")
            for lg, teams in ratings:
                L.append("### %s" % _esc(lg))
                L.append("")
                L.append("| # | Club | Attack | Defence | Rating |")
                L.append("|---:|---|---:|---:|---:|")
                for n, (t, atk, dfc, rat) in enumerate(teams, 1):
                    L.append("| %d | %s | %.2f | %.2f | **%.2f** |"
                             % (n, _esc(t), atk, dfc, rat))
                L.append("")
            L.append("Name any two clubs and a card can be produced "
                     "immediately — the engine does not need a fixture list to "
                     "price a match:")
            L.append("")
            L.append("```bash")
            L.append('python predict.py predict "Arsenal" "Chelsea" --full --best')
            L.append("```")
            L.append("")
        return _write(out_path, L)

    # ------------------------------------------------------------ the slate
    L.append("## The slate")
    L.append("")
    L.append("%d fixtures across %d competitions, strongest call first."
             % (len(rows), len({r["league"] for r in rows})))
    L.append("")
    unsched = bool(meta.get("unscheduled"))
    L.append("| %s | Competition | Match | Pick | 1 | X | 2 | O2.5 | BTTS "
             "| Score | Expected goals | Confidence |"
             % ("#" if unsched else "Kick-off"))
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|")
    for n, r in enumerate(sorted(rows, key=lambda x: -max(x["pH"], x["pD"],
                                                          x["pA"])), 1):
        when = str(n) if unsched else "%s %s" % (r["date"].strftime("%a %d %b"),
                                                 r["time"])
        L.append("| %s | %s | **%s** | %s | %s | %s | %s | %s | %s | %s | "
                 "%.2f&ndash;%.2f | %s |" % (
                     when,
                     _esc(r["league"]), _esc(r["match"]), r["pick"],
                     _pct(r["pH"]), _pct(r["pD"]), _pct(r["pA"]),
                     _pct(r["over25"]), _pct(r["btts"]), r["score"],
                     r["xgh"], r["xga"], r["confidence"]))
    L.append("")
    L.append("*Pick is the most likely 1X2 outcome. Score is the single most "
             "likely scoreline, which often disagrees with the pick — a "
             "favourite's win probability is spread across many scorelines "
             "while the draw concentrates on 1-1.*")
    L.append("")

    # -------------------------------------------------- by kick-off time
    if meta.get("unscheduled"):
        return _write(out_path, L + _tail(promoted, skipped))
    L.append("## In kick-off order")
    L.append("")
    day = None
    for r in sorted(rows, key=lambda x: (x["date"], x["time"])):
        d = r["date"].strftime("%A %d %B")
        if d != day:
            L.append("")
            L.append("### %s" % d)
            L.append("")
            day = d
        L.append("- **%s** &nbsp; %s &nbsp;&mdash;&nbsp; %s &nbsp; "
                 "`%s %s / %s / %s` &nbsp; most likely **%s** &nbsp; "
                 "(%s, xG %.2f&ndash;%.2f)" % (
                     r["time"], _esc(r["match"]), _esc(r["league"]),
                     r["pick"][0] if r["pick"] != "Draw" else "X",
                     _pct(r["pH"]), _pct(r["pD"]), _pct(r["pA"]),
                     r["score"], r["confidence"].lower(), r["xgh"], r["xga"]))
    L.append("")

    # ---------------------------------------------------------- disagreements
    if dissent:
        L.append("## Where the model disagrees with the price")
        L.append("")
        L.append("The model's own view before any blending, against the "
                 "de-vigged market. **A disagreement is the interesting part "
                 "of a forecast, not a tip** — the backtest says betting these "
                 "loses money.")
        L.append("")
        L.append("| Match | Competition | Selection | Model | Market | "
                 "Difference | Price |")
        L.append("|---|---|---|---:|---:|---:|---:|")
        for d in dissent:
            L.append("| %s | %s | %s | %s | %s | +%s | %.2f |" % (
                _esc(d["match"]), _esc(d["league"]), d["sel"],
                _pct(d["model_p"]), _pct(d["market_p"]),
                _pct(d["diff"]), d["price"]))
        L.append("")

    # ----------------------------------------------------------- caveats
    L += _cross_section(cross)
    L += _tail(promoted, skipped)
    return _write(out_path, L)


def _cross_section(cross):
    """Ties between clubs from different leagues: rated, but not priced."""
    if not cross:
        return []
    L = ["## Cross-league ties - listed, deliberately not priced", ""]
    L.append("These pair clubs from different domestic leagues, and no **measured "
             "bridge covers both sides**, so no probabilities are given. Each "
             "league is fitted separately with its own zero-sum constraint, so a "
             "Premier League attack rating of 1.4 and a Serie A one of 1.4 are "
             "not the same quantity. The country bridges measured from real UEFA "
             "and CAF club ties already price the leagues they cover "
             "('(bridged)' rows); anything listed here falls outside them.")
    L.append("")
    L.append("What follows is each club's standing **within its own league**, "
             "which is real and useful, but must not be read across the row as "
             "a head-to-head.")
    L.append("")
    L.append("| Date | Match | Home league | Home rating | Away league | "
             "Away rating |")
    L.append("|---|---|---|---:|---|---:|")
    for c in cross:
        h, a = c["sides"]
        when = (c["date"].strftime("%a %d %b") if c.get("date") is not None
                and hasattr(c["date"], "strftime") else "")
        L.append("| %s %s | **%s v %s** | %s | %.2f | %s | %.2f |" % (
            when, c["time"], _esc(h["team"]), _esc(a["team"]),
            _esc(h["league"]), h["rating"], _esc(a["league"]), a["rating"]))
    L.append("")
    L.append("*Rating is attack divided by defence within that club's own "
             "league, where 1.00 is that league's average side.*")
    L.append("")
    return L


def _tail(promoted, skipped):
    L = []
    if promoted:
        L.append("## Clubs with no record in their division")
        L.append("")
        for t in sorted(set(promoted)):
            L.append("- %s" % t)
        L.append("")
    if skipped:
        L.append("## Left out")
        L.append("")
        for x in skipped:
            L.append("- %s" % x)
        L.append("")
    L.append("---")
    L.append("")
    L.append("Ratings come from results only — goals and shots on target, "
             "time-decayed. The model has no sight of injuries, suspensions, "
             "line-ups, transfers or European fixture congestion, which is "
             "most of the reason the closing line beats it.")
    return L


def _write(path: str, lines) -> str:
    text = "\n".join(lines).rstrip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
