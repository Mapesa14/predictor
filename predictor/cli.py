"""Command line for the football score predictor."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import pandas as pd

from . import backtest, fixtures, leagues, loader, market, model, report
from .engine import (DEFAULT_EDGE_SCALE, DEFAULT_GOAL_SHRINK,
                     DEFAULT_MARKET_WEIGHT, DEFAULT_WEIGHTS,
                     DEFAULT_XI, Predictor)

from . import envfile  # noqa: E402

# A local .env, for running without docker compose; real variables still win.
envfile.load()

# FOOTBALL_DATA is what the service reads; SOCCER_DATA was the CLI's own name
# for the same folder. Honour both, service's first, so a container that sets
# one variable does not leave the CLI pointing at a Windows download path.
DEFAULT_DATA = (os.environ.get("FOOTBALL_DATA") or os.environ.get("SOCCER_DATA")
                or r"D:\Downloads July 2026\SoccerData")


def _jsonable(s):
    out = {}
    for k, v in s.items():
        if k == "correct_scores":
            out[k] = [{"score": "%d-%d" % (i, j), "p": p} for i, j, p in v]
        elif k == "ht_ft":
            out[k] = {"%s/%s" % kk: vv for kk, vv in v.items()}
        elif isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _pred(a) -> Predictor:
    """A Predictor built from the shared flags."""
    return Predictor(a.data, xi=a.xi, as_of=_asof(a),
                     goal_shrink=a.goal_shrink, edge_scale=a.edge_scale,
                     use_ladder=not a.no_ladder, weights=_weights(a),
                     market_weight=a.market_weight)


def _split_match(args):
    """Accept two team arguments, or one string with a vs/v/- separator."""
    parts = list(args)
    joined = " ".join(parts)
    low = joined.lower()
    for sep in (" vs ", " v ", " - ", " x "):
        if sep in low:
            i = low.index(sep)
            return joined[:i].strip(), joined[i + len(sep):].strip()
    if len(parts) == 2:
        return parts[0], parts[1]
    raise SystemExit('Give two teams, e.g. predict "Arsenal" "Chelsea"')


# ------------------------------------------------------------------ commands
def cmd_predict(a):
    home, away = _split_match(a.match)
    p = Predictor(a.data, xi=a.xi, as_of=_asof(a),
                  goal_shrink=a.goal_shrink, edge_scale=a.edge_scale,
                  use_ladder=not a.no_ladder, weights=_weights(a),
                  market_weight=a.market_weight)
    # A name unknown to an explicitly named league is taken as a promoted club;
    # without -l we have no league to promote it into, so it stays an error.
    s = p.predict(home, away, a.league, neutral=a.neutral,
                  allow_new=bool(a.league))
    when = _parse_when(a.date)
    if a.json:
        print(json.dumps(_jsonable(s), indent=2, default=str))
        return
    print(report.card(s["home"], s["away"], s["div"], s, when,
                      top_scores=a.scores, wide=a.full))
    # Any inexact name match is shown, so a wrong guess can never pass unseen.
    for typed, got in ((home, s["home"]), (away, s["away"])):
        if typed.strip().casefold() != got.casefold():
            print("\nRead %r as %r." % (typed.strip(), got))
    if a.best:
        print("\nBEST BETS")
        for name, prob in report.best_bets(s):
            print("  %-24s%s" % (name, report.pct(prob).rjust(6)))
    for team, isnew, src in ((s["home"], s["home_new"], s["home_source"]),
                             (s["away"], s["away_new"], s["away_source"])):
        if not isnew:
            continue
        if src == "prior":
            print("\nNote: %s has no history in %s, and none in a division we can\n"
                  "      carry a rating from. Rated with the promoted-team prior:\n"
                  "      scores 0.79x and concedes 1.16x the league average."
                  % (team, leagues.name(s["div"])))
        else:
            print("\nNote: %s has no history in %s. Rated from its %s form,\n"
                  "      shifted by the measured gap between the two divisions."
                  % (team, leagues.name(s["div"]), leagues.name(src)))
    if not (s["home_new"] or s["away_new"]) and \
            min(s["home_played"], s["away_played"]) < 12:
        print("\nNote: thin sample, %s %d matches and %s %d matches in this league."
              % (s["home"], s["home_played"], s["away"], s["away_played"]))


def cmd_slate(a):
    """Predict every fixture on the upcoming schedule."""
    p = Predictor(a.data, xi=a.xi, as_of=_asof(a),
                  goal_shrink=a.goal_shrink, edge_scale=a.edge_scale,
                  use_ladder=not a.no_ladder, weights=_weights(a),
                  market_weight=a.market_weight)
    divs = [a.league] if a.league else (leagues.TOP_10 if a.top10 else p.divs)
    rows, skipped, promoted = [], [], []
    for d in divs:
        if d not in p.divs:
            continue
        sched = p.schedule(d, a.days, a.fixtures)
        if not len(sched):
            continue
        for _, f in sched.head(a.limit).iterrows():
            try:
                s = p.predict(f["HomeTeam"], f["AwayTeam"], d, allow_new=True)
            except SystemExit as e:
                skipped.append("%s: %s v %s (%s)"
                               % (leagues.name(d), f["HomeTeam"], f["AwayTeam"], e))
                continue
            i, j, _ = s["correct_scores"][0]
            r = s["result"]
            pick = max(r, key=r.get)
            # A promoted side has no history in this division. Where it played a
            # division we also load, its real rating carries up; otherwise it
            # falls back to the measured promoted-team prior.
            newcomers = ([(s["home"], s["home_source"])] if s["home_new"] else []) + \
                        ([(s["away"], s["away_source"])] if s["away_new"] else [])
            for t, src in newcomers:
                promoted.append("%s (%s) - %s" % (
                    t, leagues.name(d),
                    "promoted-team prior" if src == "prior"
                    else "carried up from " + leagues.name(src)))
            rows.append({
                "League": leagues.name(d),
                "Date": f["Date"].strftime("%d %b") if pd.notna(f.get("Date")) else "TBD",
                "Match": "%s v %s%s" % (s["home"], s["away"], " *" if newcomers else ""),
                "Pick": {"H": "1", "D": "X", "A": "2"}[pick],
                "1": report.pct(r["H"]), "X": report.pct(r["D"]), "2": report.pct(r["A"]),
                "O2.5": report.pct(s["totals"][2.5]["over"]),
                "BTTS": report.pct(s["btts"]["yes"]),
                "Score": "%d-%d" % (i, j),
                "xG": "%.2f-%.2f" % (s["exp_home"], s["exp_away"]),
                "_conf": r[pick],
            })
    if not rows:
        print("No upcoming fixtures found. Point --fixtures at a fixtures CSV, "
              "or run: predict.py refresh-fixtures")
        return
    t = pd.DataFrame(rows).sort_values("_conf", ascending=False).drop(columns="_conf")
    if a.json:
        print(t.to_json(orient="records", indent=2))
    else:
        print(t.to_string(index=False))
        print("\n%d fixtures. Pick is the most likely 1X2 outcome, "
              "sorted by confidence." % len(t))
        if promoted:
            print("\n* No history in this division, so the rating comes from "
                  "elsewhere:")
            for t in sorted(set(promoted)):
                print("    %s" % t)
        if skipped:
            print("\n%d fixtures could not be predicted:" % len(skipped))
            for line in skipped:
                print("    %s" % line)


def cmd_brief(a):
    """Predict a date window and write it out as a PDF."""
    from . import brief, brief_md, market
    p = Predictor(a.data, xi=a.xi, as_of=_asof(a),
                  goal_shrink=a.goal_shrink, edge_scale=a.edge_scale,
                  use_ladder=not a.no_ladder, weights=_weights(a),
                  market_weight=a.market_weight)
    # Named pairings bypass the schedule entirely: the engine can price any two
    # clubs without a fixture list, which is the only option when the feed is
    # down or the competition has no published schedule yet.
    if a.pair or a.top_pairs:
        return _brief_pairs(a, p)

    cache = os.path.join(a.data, "fixtures.csv")
    fx = fixtures.load_any(a.fixtures, cache)
    if not len(fx):
        raise SystemExit("No fixtures file. Run: predict.py refresh-fixtures")

    start = _parse_when(a.start) if a.start else datetime.now()
    end = _parse_when(a.end) if a.end else (start + pd.Timedelta(days=2))
    end = end.replace(hour=23, minute=59)
    fx = fx[(fx["Date"] >= pd.Timestamp(start).normalize()) &
            (fx["Date"] <= pd.Timestamp(end))]
    if a.after:                       # keep only kick-offs at or past this time
        hh = fx["Time"].fillna("00:00").astype(str)
        same_day = fx["Date"].dt.normalize() == pd.Timestamp(start).normalize()
        fx = fx[~same_day | (hh >= a.after)]
    if a.league:
        fx = fx[fx["Div"] == a.league]
    elif a.top10:
        fx = fx[fx["Div"].isin(leagues.TOP_10)]
    fx = fx.sort_values(["Date", "Time", "Div"])
    if not len(fx) and not str(a.out or "").lower().endswith(".md"):
        raise SystemExit("No fixtures in that window.")

    rows, dissent, skipped, promoted, unloaded = [], [], [], [], set()
    cross, caf_rows = [], []
    for _, f in fx.iterrows():
        d = f["Div"]
        if d == "CAFCL":
            # CAF ties go through the African bridge; a club from a federation
            # with no loaded league is rated at federation level.
            try:
                s = p.predict_caf(f["HomeTeam"], f["AwayTeam"])
            except SystemExit as e:
                skipped.append("%s v %s (%s)" % (f["HomeTeam"], f["AwayTeam"], e))
                continue
            caf_rows.append(_caf_row(s, f, a.tz))
            continue
        if d not in p.divs and _is_domestic_code(d):
            # A domestic league we simply do not load, such as Serie B. Its
            # clubs are not a cross-league tie just because some of them were
            # relegated from a league we do load - leave the whole division out.
            unloaded.add(d)
            continue
        if d not in p.divs:
            # A competition we do not model, such as a continental cup. With a
            # measured country bridge these can be priced properly; without one
            # the clubs' domestic ratings are reported and left unpriced.
            if p.can_bridge(f["HomeTeam"], f["AwayTeam"]):
                try:
                    s = p.predict_cross(f["HomeTeam"], f["AwayTeam"], comp=d)
                except SystemExit as e:
                    skipped.append("%s v %s (%s)"
                                   % (f["HomeTeam"], f["AwayTeam"], e))
                    continue
                rows.append(_row(s, "%s (bridged)" % d, f, a.tz))
                continue
            tie = _cross_tie(p, f)
            if tie and tie.get("missing"):
                skipped.append(
                    "%s v %s (%s): no rating for %s - that league is not loaded"
                    % (f["HomeTeam"], f["AwayTeam"], d,
                       " and ".join(tie["missing"])))
            elif tie:
                cross.append(tie)
            else:
                unloaded.add(d)
            continue
        try:
            s = p.predict(f["HomeTeam"], f["AwayTeam"], d, allow_new=True, odds=f)
        except SystemExit as e:
            skipped.append("%s v %s (%s)" % (f["HomeTeam"], f["AwayTeam"], e))
            continue
        r = s["result"]
        pick = max(r, key=r.get)
        i, j, _ = s["correct_scores"][0]
        is_cup = isinstance(f.get("Comp"), str) and f.get("Comp").strip() != ""
        for team, isnew, src in ((s["home"], s["home_new"], s["home_source"]),
                                 (s["away"], s["away_new"], s["away_source"])):
            # A cup tie priced on the higher division's scale carries the lower
            # club in, but that club is not "with no history in its division":
            # listing Reading that way read as though it had been promoted.
            if isnew and not is_cup:
                promoted.append("%s (%s, %s)" % (
                    team, leagues.name(d),
                    "no lower-division record, promoted-team prior used"
                    if src == "prior" else "rated from " + leagues.name(src)))
        conf = r[pick]
        when, time_ = _local_kickoff(f, a.tz)
        rows.append({
            "league": _comp_label(f, d),
            "time": time_,
            "date": when,
            "match": "%s v %s" % (s["home"], s["away"]),
            "pH": r["H"], "pD": r["D"], "pA": r["A"],
            "pick": {"H": "Home", "D": "Draw", "A": "Away"}[pick],
            "over25": s["totals"][2.5]["over"], "btts": s["btts"]["yes"],
            "score": "%d-%d" % (i, j),
            "xgh": s["exp_home"], "xga": s["exp_away"],
            "confidence": _confidence(conf),
        })
        # the model's own dissent from the price, before blending
        if s.get("market_used"):
            mp, kp = s["model_result"], s["market_result"]
            for sel, key, ocol in (("Home", "H", "AvgH"), ("Draw", "D", "AvgD"),
                                   ("Away", "A", "AvgA")):
                diff = mp[key] - kp[key]
                price = f.get(ocol)
                if diff > 0.06 and pd.notna(price):
                    dissent.append({
                        "match": "%s v %s" % (s["home"], s["away"]),
                        "league": leagues.name(d), "sel": sel,
                        "model_p": mp[key], "market_p": kp[key],
                        "diff": diff, "price": float(price)})

    out = a.out or "predictions.pdf"
    as_md = out.lower().endswith(".md")
    if not rows and not cross and not caf_rows and not as_md:
        raise SystemExit("Nothing could be predicted in that window.")
    dissent.sort(key=lambda x: -x["diff"])
    if unloaded:
        counts = fx[fx["Div"].isin(unloaded)].groupby("Div").size()
        skipped.append("divisions not loaded, so not predicted: " + ", ".join(
            "%s (%d)" % (_UNLOADED_NAMES.get(d, d), int(counts.get(d, 0)))
            for d in sorted(unloaded)))

    same_day = pd.Timestamp(start).normalize() == pd.Timestamp(end).normalize()
    label = (pd.Timestamp(start).strftime("%A %d %B %Y") if same_day else
             "%s to %s" % (pd.Timestamp(start).strftime("%a %d %b %Y"),
                           pd.Timestamp(end).strftime("%a %d %b %Y")))
    if a.after and same_day:
        label += ", kick-offs from %s" % a.after
    meta = {"n_matches": format(len(p.df), ","),
            "market_weight": a.market_weight,
            "tz_label": _tz_label(a.tz),
            "notes": list(a.note or [])}
    if as_md:
        brief_md.build(rows, out, label, datetime.now(), meta,
                       dissent=dissent[:a.dissent], skipped=skipped,
                       promoted=promoted, empty_note=_empty_note(p, start, end),
                       ratings=None if rows else _ratings(p, a), cross=cross)
    else:
        brief.build(rows, out, label, datetime.now(), meta,
                    dissent=dissent[:a.dissent], skipped=skipped,
                    promoted=promoted, cross=cross, caf=caf_rows)
    print("Wrote %s" % out)
    if caf_rows:
        print("  %d CAF tie(s) priced through the African bridge" % len(caf_rows))
    if cross:
        print("  %d cross-league tie(s) listed with ratings but not priced"
              % len(cross))
    if not rows:
        print("  no fixtures in that window - see the file for why")
        return
    print("  %d fixtures across %d competitions"
          % (len(rows), len({r["league"] for r in rows})))
    print("  %d model/price disagreements listed" % len(dissent[:a.dissent]))
    if skipped:
        print("  %d note(s) on what was left out" % len(skipped))


def _empty_note(p, start, end):
    """Explain an empty window from the calendar in the data, not from a guess."""
    d = p.df[p.df["Div"].isin(leagues.TOP_10)]
    lo = pd.Timestamp(start).dayofyear - 1
    hi = pd.Timestamp(end).dayofyear + 1
    same = d[(d["Date"].dt.dayofyear >= lo) & (d["Date"].dt.dayofyear <= hi)]
    years = sorted(set(d["Date"].dt.year))
    counts = same.groupby(same["Date"].dt.year).size().to_dict()
    lines = ["The fixtures feed on disk covers **%s** and holds nothing for "
             "this window. The live feed at football-data.co.uk is returning "
             "HTTP 503 right now, so it could not be refreshed." % _fx_span(p)]
    if years:
        lines.append("")
        lines.append("That is very likely correct rather than a gap in the "
                     "data. Across the same three days of previous seasons, "
                     "these ten leagues played:")
        lines.append("")
        lines.append("| Season | Matches on these dates |")
        lines.append("|---|---:|")
        for y in years:
            lines.append("| %d | %d |" % (y, counts.get(y, 0)))
        lines.append("")
        if max(counts.values(), default=0) < 10:
            lines.append("Early September is the FIFA international window and "
                         "Europe's domestic leagues pause through it. This "
                         "engine rates club sides only — it has no "
                         "international-team model — so there is nothing today "
                         "it can honestly forecast.")
            lines.append("")
            lines.append("Club football in these leagues resumes the following "
                         "weekend. Re-run this once the feed is back:")
            lines.append("")
            lines.append("```bash")
            lines.append("python predict.py refresh-fixtures && \\")
            lines.append("  python predict.py brief --top10 -o slate.md")
            lines.append("```")
    return "\n".join(lines)


def _parse_pair(text):
    """'N1:Ajax v PSV Eindhoven' -> ('N1', 'Ajax', 'PSV Eindhoven')."""
    if ":" not in text:
        raise SystemExit("Pair needs a division, e.g. --pair \"N1:Ajax v PSV\"")
    div, rest = text.split(":", 1)
    low = rest.lower()
    for sep in (" v ", " vs ", " - "):
        if sep in low:
            i = low.index(sep)
            return div.strip(), rest[:i].strip(), rest[i + len(sep):].strip()
    raise SystemExit("Could not read a fixture from %r" % text)


def _brief_pairs(a, p):
    """Price named pairings and render them, with no schedule involved."""
    from . import brief, brief_md
    wanted = list(a.pair or [])
    if a.top_pairs:
        # every ordered pairing among a league's strongest clubs, which is the
        # closest thing to a schedule when no schedule is published
        for d in _divs_of(a, p):
            table = model.strength_table(p.models(d)["FT"])[:a.top_pairs]
            names = [t for t, *_ in table]
            wanted += ["%s:%s v %s" % (d, h, x)
                       for h in names for x in names if h != x]
    rows, promoted, skipped = [], [], []
    for text in wanted:
        div, home, away = _parse_pair(text)
        if div not in p.divs:
            skipped.append("%s: no data loaded for division %s" % (text, div))
            continue
        try:
            s = p.predict(home, away, div, allow_new=True)
        except SystemExit as e:
            skipped.append("%s (%s)" % (text, e))
            continue
        r = s["result"]
        pick = max(r, key=r.get)
        i, j, _ = s["correct_scores"][0]
        for team, isnew, src in ((s["home"], s["home_new"], s["home_source"]),
                                 (s["away"], s["away_new"], s["away_source"])):
            if isnew:
                promoted.append("%s (%s, %s)" % (
                    team, leagues.name(div),
                    "no lower-division record, promoted-team prior used"
                    if src == "prior" else "rated from " + leagues.name(src)))
        rows.append({
            "league": leagues.label(div), "time": "-",
            "date": pd.Timestamp(datetime.now().date()),
            "match": "%s v %s" % (s["home"], s["away"]),
            "pH": r["H"], "pD": r["D"], "pA": r["A"],
            "pick": {"H": "Home", "D": "Draw", "A": "Away"}[pick],
            "over25": s["totals"][2.5]["over"], "btts": s["btts"]["yes"],
            "score": "%d-%d" % (i, j),
            "xgh": s["exp_home"], "xga": s["exp_away"],
            "confidence": _confidence(r[pick]),
        })
    if not rows:
        raise SystemExit("None of those pairings could be priced.\n  "
                         + "\n  ".join(skipped))

    out = a.out or "pairings.pdf"
    meta = {"n_matches": format(len(p.df), ","),
            "market_weight": 0.0,          # no prices exist for a non-fixture
            "unscheduled": True}
    label = ("Head-to-head pairings, priced %s - these are ratings of the "
             "matchup, not scheduled fixtures"
             % datetime.now().strftime("%d %b %Y"))
    if out.lower().endswith(".md"):
        brief_md.build(rows, out, label, datetime.now(), meta,
                       skipped=skipped, promoted=promoted)
    else:
        brief.build(rows, out, label, datetime.now(), meta,
                    skipped=skipped, promoted=promoted)
    print("Wrote %s" % out)
    print("  %d pairings across %d competitions"
          % (len(rows), len({r["league"] for r in rows})))
    for line in skipped:
        print("  left out: %s" % line)


def _divs_of(a, p):
    """Divisions selected by -l (comma separated), --top10, or everything."""
    if a.league:
        return [d.strip() for d in a.league.split(",") if d.strip() in p.divs]
    return [d for d in (leagues.TOP_10 if getattr(a, "top10", False) else p.divs)
            if d in p.divs]


def _cross_tie(p, f):
    """Ratings for a tie between clubs from two different leagues.

    Each league is fitted with its own zero-sum constraint, so an English
    attack rating and an Italian one are not on the same scale. Until a bridge
    between leagues is measured, these are reported side by side and left
    unpriced rather than guessed at.
    """
    import math
    out = {"comp": str(f.get("Div", "")), "sides": [], "missing": []}
    for side in ("HomeTeam", "AwayTeam"):
        name = str(f[side]).strip()
        try:
            team, div = p.resolve(name, None, fuzzy=False)
        except SystemExit:
            out["missing"].append(name)
            continue
        m = p.models(div)["FT"]
        atk, dfc = m._team(team)
        out["sides"].append({
            "team": team, "div": div, "league": leagues.name(div),
            "attack": math.exp(atk),
            "defence": math.exp(dfc),
            "rating": math.exp(atk - dfc)})
    if out["missing"]:
        return out                        # caller reports which club is unrated
    if out["sides"][0]["div"] == out["sides"][1]["div"]:
        return None                       # same league: it can be priced normally
    out["date"] = f.get("Date")
    out["time"] = str(f["Time"])[:5] if pd.notna(f.get("Time")) else "TBD"
    return out


def _ratings(p, a, top=6):
    """Current strength table per league, for a brief with nothing to forecast."""
    divs = _divs_of(a, p)
    out = []
    for d in divs:
        try:
            m = p.models(d)["FT"]
        except SystemExit:
            continue
        out.append((leagues.label(d),
                    [(t, atk, dfc, rat)
                     for t, atk, dfc, rat, _ in model.strength_table(m)[:top]]))
    return out


def _fx_span(p):
    cache = os.path.join(p.root, "fixtures.csv")
    try:
        fx = fixtures.from_csv(cache)
        return "%s to %s" % (fx["Date"].min().date(), fx["Date"].max().date())
    except Exception:
        return "an unknown range"


# football-data.co.uk division codes for domestic leagues this build does not
# load, named so the note reads as leagues rather than codes.
_UNLOADED_NAMES = {
    "I2": "Serie B", "SP2": "Segunda Division", "D2": "2. Bundesliga",
    "F2": "Ligue 2", "SC1": "Scottish Championship", "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
}


def _comp_label(f, div: str) -> str:
    """The competition a row belongs to.

    A cup tie is priced on a league division's scale, so its row carries that
    division's code - but heading it "Premier League (England)" would tell the
    reader it is a league match. A fixtures file may name the competition in a
    `Comp` column; otherwise the division's own label is used.
    """
    comp = f.get("Comp")
    if isinstance(comp, str) and comp.strip():
        return "%s (%s)" % (comp.strip(), leagues.country(div))
    return leagues.label(div)


def _is_domestic_code(div: str) -> bool:
    """A football-data league code (E0, SC2, I2...), as opposed to a cup."""
    import re
    return bool(re.fullmatch(r"(E[0-3C]|SC[0-3]|[A-Z]{1,3}\d)", str(div)))


_TZ_LABELS = {"Africa/Dar_es_Salaam": "EAT", "Europe/London": "UK",
              "Africa/Nairobi": "EAT", "UTC": "UTC"}


def _tz_label(tz):
    return _TZ_LABELS.get(tz, tz)


def _local_kickoff(f, tz):
    """(date, 'HH:MM') in the reader's time zone, from a fixture row.

    Each row's time is read in its own league's source zone - UK for the
    football-data feed, East Africa for the Tanzanian league site - via
    `leagues.source_tz`, the same rule the service uses. Reading every row as
    UK time put tonight's 19:00 Dar es Salaam kick-off in a brief at 21:00.
    Converting the full timestamp rather than adding hours means a late UK
    kick-off correctly lands on the next calendar day in East Africa.
    """
    note = f.get("WhenNote")
    if isinstance(note, str) and note.strip():
        return note.strip(), "TBC"
    d = pd.Timestamp(f["Date"])
    t = f.get("Time")
    if not isinstance(t, str) or ":" not in t:
        return d, "TBC"
    hh, mm = t.strip()[:5].split(":")
    ts = d.replace(hour=int(hh), minute=int(mm))
    try:
        ts = (ts.tz_localize(leagues.source_tz(f.get("Div")))
              .tz_convert(tz).tz_localize(None))
    except Exception:
        pass
    return ts, ts.strftime("%H:%M")


def _caf_row(s, f, tz):
    """A CAF tie: the match itself, what fed each rating, and the tie odds."""
    from .engine import tie_outcome
    r = s["result"]
    pick = max(r, key=r.get)
    i, j, _ = s["correct_scores"][0]
    when, time_ = _local_kickoff(f, tz)

    def basis(fed, source, ties):
        if not fed:
            return source
        return "%s (%s CAF ties)" % (source.replace("federation ", "") + " fed.",
                                     ties if ties else "no")

    row = {
        "date": when, "time": time_,
        "match": "%s v %s" % (s["home"], s["away"]),
        "pH": r["H"], "pD": r["D"], "pA": r["A"],
        "pick": {"H": "Home", "D": "Draw", "A": "Away"}[pick],
        "over25": s["totals"][2.5]["over"], "btts": s["btts"]["yes"],
        "score": "%d-%d" % (i, j),
        "xgh": s["exp_home"], "xga": s["exp_away"],
        "confidence": _confidence(r[pick]),
        "home_basis": basis(s["home_fed"], s["home_source"], s["home_fed_ties"]),
        "away_basis": basis(s["away_fed"], s["away_source"], s["away_fed_ties"]),
        "thin": (s["home_fed"] and not s["home_fed_ties"]) or
                (s["away_fed"] and not s["away_fed_ties"]),
        "leg1": None, "tie": None,
    }
    lh, la = f.get("Leg1H"), f.get("Leg1A")
    if pd.notna(lh) and pd.notna(la):
        row["leg1"] = (int(lh), int(la))
        row["tie"] = tie_outcome(s["matrix"], int(lh), int(la))
    return row


def _row(s, league_label, f, tz="Africa/Dar_es_Salaam"):
    """One slate row from a prediction summary."""
    r = s["result"]
    pick = max(r, key=r.get)
    i, j, _ = s["correct_scores"][0]
    when, time_ = _local_kickoff(f, tz)
    return {
        "league": league_label,
        "time": time_,
        "date": when,
        "match": "%s v %s" % (s["home"], s["away"]),
        "pH": r["H"], "pD": r["D"], "pA": r["A"],
        "pick": {"H": "Home", "D": "Draw", "A": "Away"}[pick],
        "over25": s["totals"][2.5]["over"], "btts": s["btts"]["yes"],
        "score": "%d-%d" % (i, j),
        "xgh": s["exp_home"], "xga": s["exp_away"],
        "confidence": _confidence(r[pick]),
    }


def _confidence(p):
    """Plain words for how strong a call is, so nobody reads 38% as a tip."""
    if p >= 0.65:
        return "Strong"
    if p >= 0.50:
        return "Clear"
    if p >= 0.42:
        return "Slight lean"
    return "Close to a coin toss"


def cmd_table(a):
    p = Predictor(a.data, xi=a.xi, as_of=_asof(a),
                  goal_shrink=a.goal_shrink, edge_scale=a.edge_scale,
                  use_ladder=not a.no_ladder, weights=_weights(a),
                  market_weight=a.market_weight)
    divs = [a.league] if a.league else (leagues.TOP_10 if a.top10 else p.divs)
    for d in divs:
        if d not in p.divs:
            continue
        m = p.models(d)["FT"]
        print("\n%s  -  %d matches, home advantage %+.3f, rho %+.3f"
              % (leagues.label(d), m.n_matches, m.home_adv, m.rho))
        print("%-24s%8s%8s%9s%7s" % ("Team", "Attack", "Defence", "Rating", "Pld"))
        print("-" * 56)
        for t, atk, dfc, rating, pld in model.strength_table(m):
            print("%-24s%8.2f%8.2f%9.2f%7d" % (t, atk, dfc, rating, pld))


def cmd_fixtures(a):
    p = Predictor(a.data, xi=a.xi, as_of=_asof(a),
                  goal_shrink=a.goal_shrink, edge_scale=a.edge_scale,
                  use_ladder=not a.no_ladder, weights=_weights(a),
                  market_weight=a.market_weight)
    divs = [a.league] if a.league else (leagues.TOP_10 if a.top10 else p.divs)
    for d in divs:
        if d not in p.divs:
            continue
        s = p.schedule(d, a.days, a.fixtures)
        print("\n%s: %d fixtures" % (leagues.label(d), len(s)))
        if not len(s):
            print("  Every pairing has already been played, so the remaining "
                  "schedule cannot be derived offline.")
            if d in fixtures.SPLIT_LEAGUES:
                print("  This league has a split second phase. Use --fixtures "
                      "or refresh-fixtures.")
            continue
        for _, f in s.head(a.limit).iterrows():
            when = f["Date"].strftime("%a %d %b") if pd.notna(f.get("Date")) else "TBD"
            print("  %-12s%s v %s" % (when, f["HomeTeam"], f["AwayTeam"]))
        if len(s) > a.limit:
            print("  ... and %d more" % (len(s) - a.limit))


def cmd_refresh(a):
    dest = os.path.join(a.data, "fixtures.csv")
    print("Downloading %s -> %s" % (fixtures.FEED_URL, dest))
    fixtures.refresh(dest)
    fx = fixtures.from_csv(dest)
    print("Saved %d fixtures, %s to %s"
          % (len(fx), fx["Date"].min().date(), fx["Date"].max().date()))


def cmd_backtest(a):
    df = loader.load(a.data)
    divs = [a.league] if a.league else leagues.TOP_10
    parts = []
    for d in divs:
        if d not in set(df["Div"]):
            continue
        bt = backtest.walk_forward(df, d, xi=a.xi, min_train=a.min_train,
                                   refit_days=a.refit_days,
                                   goal_shrink=a.goal_shrink,
                                   edge_scale=a.edge_scale,
                                   weights=_weights(a),
                                   market_weight=a.market_weight)
        if not len(bt):
            continue
        s = backtest.score(bt)
        print("%-26s n=%-5d logloss %.4f  rps %.4f  acc %.3f  O2.5 %.4f  BTTS %.4f"
              % (leagues.name(d), s["n"], s["logloss_1x2"], s["rps"], s["acc"],
                 s["logloss_ou25"], s["logloss_btts"]))
        parts.append(bt)
    if not parts:
        raise SystemExit("Nothing to backtest.")
    allbt = pd.concat(parts, ignore_index=True)
    s = backtest.score(allbt)
    print("-" * 100)
    print("%-26s n=%-5d logloss %.4f  rps %.4f  acc %.3f  O2.5 %.4f  BTTS %.4f"
          % ("ALL", s["n"], s["logloss_1x2"], s["rps"], s["acc"],
             s["logloss_ou25"], s["logloss_btts"]))
    if "logloss_market" in s:
        print("%-26s n=%-5d logloss %.4f  rps %.4f   (bookmaker consensus)"
              % ("MARKET BASELINE", s["n_with_odds"], s["logloss_market"],
                 s["rps_market"]))

    print("\nCALIBRATION - Over 2.5 goals")
    print(backtest.calibration(allbt, "pOver25", "over25").to_string(index=False))
    print("\nCALIBRATION - home win")
    hw = allbt.assign(_hw=(allbt["FTR"] == "H").astype(int))
    print(backtest.calibration(hw, "pH", "_hw").to_string(index=False))

    v = backtest.value_bets(allbt, edge=a.edge)
    if len(v):
        n, wins = len(v), int(v["won"].sum())
        print("\nVALUE BETS at >= %.0f%% edge vs closing consensus" % (100 * a.edge))
        print("  selections %d, won %d (%.1f%%)" % (n, wins, 100 * wins / n))
        print("  flat stake ROI  %+.2f%%" % (100 * v["flat_pnl"].mean()))
        print("  Kelly ROI       %+.2f%% of staked"
              % (100 * v["kelly_pnl"].sum() / max(v["kelly_stake"].sum(), 1e-9)))
    if a.out:
        allbt.to_csv(a.out, index=False)
        print("\nWrote %s" % a.out)


def cmd_form(a):
    import math
    p = Predictor(a.data, xi=a.xi, as_of=_asof(a),
                  goal_shrink=a.goal_shrink, edge_scale=a.edge_scale,
                  use_ladder=not a.no_ladder, weights=_weights(a),
                  market_weight=a.market_weight)
    t, d = p.resolve(" ".join(a.team), a.league)
    m = p.form(t, d, a.n)
    print("%s - %s, last %d" % (t, leagues.label(d), len(m)))
    for _, r in m.iterrows():
        mark = "H" if r["HomeTeam"] == t else "A"
        print("  %s  %-3s %-18s %d-%d  %-18s"
              % (r["Date"].strftime("%d %b %y"), mark, r["HomeTeam"],
                 r["FTHG"], r["FTAG"], r["AwayTeam"]))
    mm = p.models(d)["FT"]
    print("\nRatings: attack %.2f  defence %.2f"
          % (math.exp(mm.attack[t]), math.exp(mm.defence[t])))


def cmd_refresh_results(a):
    """Refresh the current season's results from football-data, in place."""
    from . import refresh
    divs = a.divs or refresh.DIVS
    rows = refresh.refresh_footballdata(a.data, divs, since=a.since)
    print("%-5s %-10s %8s %8s  %s"
          % ("Div", "Wrote to", "Added", "Total", "Note"))
    print("-" * 74)
    for r in rows:
        print("%-5s %-10s %8d %8d  %s" % (r["div"], r["file"], r["added"],
                                           r["total"], r["note"]))
    ok = [r for r in rows if r["note"] == ""]
    print("\n%d divisions up to date; add the rest with --divs. "
          "Fixtures: run refresh-fixtures." % len(ok))


def cmd_update(a):
    """Everything online in one go: results then fixtures. Additive only."""
    from . import refresh
    rows = refresh.refresh_all(a.data, since=a.since)
    print("%-5s %-10s %8s %8s  %s"
          % ("Div", "Wrote to", "Added", "Total", "Note"))
    print("-" * 74)
    for r in rows:
        print("%-5s %-10s %8d %8d  %s" % (r["div"], r["file"], r["added"],
                                           r["total"], r["note"]))
    print("\nDone. The engine re-fits from this pool on its next load - "
          "no model file to retrain.")


def cmd_refresh_openfootball(a):
    """Second source: add / refresh an African division from openfootball."""
    from . import refresh
    raw = a.raw
    out = a.outdir
    os.makedirs(raw, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    for div in a.leagues or refresh.OPENFOOTBALL_WORLD:
        try:
            r = refresh._latest_openfootball(raw, out, div)
        except ValueError as e:
            print("skip %s: %s" % (div, e))
            continue
        print("%s <- %s" % (div, r["source"]))
        print("   wrote %d rows to %s.csv" % (r["written"].get(div, 0), div))
        if div in r["merges"]:
            print("   merges: %s" % "; ".join(
                "%s -> %s" % (", ".join(v[1:]), v[0])
                for v in r["merges"][div].items()))
    print("\nThese leagues live in the repo bundle; the engine merges them "
          "with the football-data pool automatically.")


def cmd_refresh_tanzania(a):
    """Third source: the NBC Premier League, from the league's own site.

    Neither feed carries it - openfootball stops in June 2026 and football-data
    has never covered Africa - so without this the home league is the one
    competition in the product with no dates and no kick-off times.
    """
    from . import adapters, tanzania
    try:
        r = tanzania.sync(a.root, url=a.url)
    except ValueError as e:
        print("tanzania: %s" % e)
        return
    print("%s  %d events read from %s" % (tanzania.DIV, r["events"], a.url))
    print("   results  %3d on file (%d published on the page) -> %s"
          % (r["results"], r["scraped_results"], r["results_file"]))
    print("   fixtures %3d, next %s -> %s"
          % (r["fixtures"], r["next_fixture"], r["fixtures_file"]))
    written, _ = adapters.build_csvs(a.raw, a.outdir)
    print("   rebuilt %s.csv: %d rows, latest result %s"
          % (tanzania.DIV, written.get(tanzania.DIV, 0), r["latest_result"]))


def _slate_rows(p, days: int):
    """The coming fixtures, shaped exactly as the service's slate shapes them.

    Only fixtures with a published kick-off: the record's whole claim is that
    the prediction predates the match, and without a time there is nothing to
    predate.
    """
    rows = []
    for d in p.divs:
        sched = p.schedule(d, days)
        if not len(sched) or "Date" not in sched.columns:
            continue
        real = sched[sched["Date"].notna()]
        if "Time" not in real.columns:
            continue
        real = real[real["Time"].notna()]
        for _, f in real.iterrows():
            odds = f if pd.notna(f.get("AvgH")) else None
            try:
                s = p.predict(f["HomeTeam"], f["AwayTeam"], d,
                              allow_new=True, odds=odds)
            except SystemExit:
                continue
            i, j, _ = s["correct_scores"][0]
            r = s["result"]
            mk = None
            if odds is not None:
                try:
                    q = market.devig([float(f["AvgH"]), float(f["AvgD"]),
                                      float(f["AvgA"])], "shin")
                    mk = {"1": float(q[0]), "X": float(q[1]), "2": float(q[2])}
                except (ValueError, TypeError, KeyError):
                    mk = None
            ko = leagues.kickoff(d, f["Date"], f.get("Time"))
            if ko is None:
                continue
            rows.append({
                "div": d, "league": leagues.name(d),
                "date": ko.isoformat(),
                "home": s["home"], "away": s["away"],
                "p": {"1": r["H"], "X": r["D"], "2": r["A"]},
                "market": mk,
                "o25": s["totals"][2.5]["over"], "btts": s["btts"]["yes"],
                "score": "%d-%d" % (i, j),
                "xg": "%.2f-%.2f" % (s["exp_home"], s["exp_away"]),
            })
    return rows


def cmd_record_publish(a):
    """Freeze the coming fixtures into the append-only record."""
    from . import record
    p = _pred(a)
    rows = _slate_rows(p, a.days)
    r = record.publish(rows, a.repo)
    print("wrote %d new prediction(s) to %s" % (r["written"], r["file"]))
    print("   already recorded  %d" % r["already_recorded"])
    print("   refused, started  %d" % r["refused_already_started"])
    print("   refused, no time  %d" % r["refused_no_kickoff"])
    print("   total on file     %d" % r["total"])
    print("\nA row is never rewritten. Re-run this as often as you like.")


def cmd_record(a):
    """Show the record: what was predicted, and what happened."""
    from . import record
    p = _pred(a)
    s = record.summary(a.repo, p.df)
    if not s["published"]:
        print("Nothing published yet. Run `record-publish` before kick-off; "
              "the record only counts what was written down first.")
        return
    print("Published %d  ·  settled %d  ·  pending %d  ·  since %s"
          % (s["published"], s["settled"], s["pending"],
             (s["first_published"] or "")[:10]))
    c = s["chain"]
    print("Chain: %s (%s)" % ("intact" if c["ok"] else "BROKEN", c["note"]))
    o, m = s["overall"], s["market"]
    if o:
        print("\n%-22s %8s %8s %8s" % ("", "log-loss", "RPS", "hit rate"))
        print("%-22s %8.4f %8.4f %7.1f%%"
              % ("model", o["logloss_1x2"], o["rps"], 100 * o["acc"]))
        if m:
            print("%-22s %8.4f %8.4f %7.1f%%   (on the %d rows with a price)"
                  % ("closing price", m["logloss_market"], m["rps_market"],
                     100 * m["acc_market"], m["n_with_price"]))
            print("%-22s %8.4f" % ("model, same rows",
                                   m["logloss_model_same_rows"]))
    if s["by_league"]:
        print("\n%-6s%-26s%7s%10s%9s" % ("Code", "League", "N", "log-loss", "Hit"))
        print("-" * 60)
        for r in s["by_league"][:15]:
            print("%-6s%-26s%7d%10.4f%8.1f%%"
                  % (r["div"], r["league"][:25], r["n"], r["logloss"],
                     100 * r["acc"]))
    if s["calibration"]:
        print("\nCalibration on the favourite")
        print("%-16s%7s%12s%11s" % ("Band", "N", "Predicted", "Realised"))
        print("-" * 46)
        for b in s["calibration"]:
            print("%-16s%7d%11.1f%%%10.1f%%"
                  % (b["bin"], b["n"], 100 * b["predicted"],
                     100 * b["realised"]))


def cmd_record_verify(a):
    from . import record
    r = record.verify(a.repo)
    print(("OK   " if r["ok"] else "FAIL ") + r["note"])
    print("%d row(s) on file" % r["rows"])


def cmd_record_migrate(a):
    """Copy the file record into DATABASE_URL, keeping its hash chain intact.

    Run once when switching storage. Without it the database starts empty and
    publishing restarts the chain, orphaning everything already published.
    """
    from . import db, record
    if db.url() is None:
        print("Set DATABASE_URL first - there is no database to migrate into.")
        return
    try:
        r = record.migrate(a.repo)
    except RuntimeError as e:
        print("refused: %s" % e)
        return
    print(("OK   " if r["ok"] else "FAIL ") + r["reason"])
    print("%d row(s) copied, %d now in the database"
          % (r["migrated"], r.get("rows", 0)))


def cmd_live_leagues(a):
    """Look up API-Football league ids. Spends one request of the live budget.

    Admin only, and deliberately a CLI command rather than an HTTP route: as a
    public endpoint any visitor could spend the day's quota with it.
    """
    from service import live
    if not live.configured():
        print("Set LIVE_API_KEY first - this asks API-Football directly.")
        return
    try:
        rows = live.find_leagues(" ".join(a.query))
    except RuntimeError as e:
        print("refused: %s" % e)
        return
    if not rows:
        print("no leagues match %r" % " ".join(a.query))
    else:
        print("%-8s%-34s%-10s%s" % ("Id", "League", "Type", "Country"))
        print("-" * 64)
        for r in rows:
            print("%-8s%-34s%-10s%s" % (r["id"], (r["name"] or "")[:33],
                                        r["type"] or "", r["country"] or ""))
    b = live.budget(live.default_store())
    print("\nlive budget: %d of %d available in the rolling 24h (%d held back)"
          % (b["available"], b["daily_limit"], b["reserve"]))
    print("Put the ids you want in LIVE_LEAGUES, or correct DIV_LEAGUES in "
          "service/live.py.")


def cmd_live_status(a):
    """The live-score budget, without spending any of it."""
    from service import live
    s = live.status()
    if not s.get("enabled"):
        print("Live scores are off: LIVE_API_KEY is not set on the service.")
        return
    if s.get("error"):
        print("live store unavailable: %s" % s["error"])
        return
    b = s["budget"]
    print("used in the last 24h  %d of %d  (reserve %d)"
          % (b["used_24h"], b["daily_limit"], b["reserve"]))
    print("available now         %d" % b["available"])
    print("provider says left    %s" % (b["provider_remaining"]
                                          if b["provider_remaining"] is not None
                                          else "not reported yet"))
    print("last minute           %d of %d" % (b["used_last_minute"],
                                              b["minute_limit"]))
    print("cooling down until    %s" % (b["cooling_down_until"] or "-"))
    print("last attempt          %s" % (s["last_attempt"] or "never"))
    print("last result           %s" % (s.get("last_result") or "-"))
    print("snapshot age          %s" % ("%ds" % s["snapshot_age_s"]
                                          if s["snapshot_age_s"] is not None
                                          else "no snapshot yet"))


def cmd_refresh_fixtures_api(a):
    """Fixtures for every league and cup the feed has not published, from
    API-Football. One request per day fetched, through the live-score budget."""
    from service import fixtures_api, live
    if not live.configured():
        print("Set LIVE_API_KEY first - this asks API-Football directly.")
        return
    p = _pred(a)
    r = fixtures_api.sync(a.root, p, days=a.days)
    print("fetched: %s" % (", ".join(r["fetched"]) or "nothing"))
    for e in r["errors"]:
        print("   error: " + e)
    print("wrote %d fixtures (%d league, %d cup) -> %s"
          % (r["rows"], r["leagues"], r["cups"], r["file"]))
    for key, title in (
            ("unresolved", "names not matched - add to fixtures_api.ALIASES only "
                           "when sure which club it is"),
            ("not_loaded", "divisions not in our data"),
            ("stale", "leagues whose results in our data are stale"),
            ("no_gap", "cup ties across divisions with no measured gap")):
        items = r["report"].get(key) or []
        if items:
            print("\n%s (%d):" % (title, len(items)))
            for x in items[:40]:
                print("   " + x)


def cmd_tips(a):
    """The coming clear picks: short list, long list, and what to avoid."""
    from . import tips
    p = _pred(a)
    out = tips.build(_slate_rows(p, a.days))

    def show(title, rows, dc=False):
        print("\n%s (%d)" % (title, len(rows)))
        if not rows:
            print("   none")
            return
        for c in rows:
            ko = (c["kickoff"] or "")[:16].replace("T", " ")
            match = (str(c["home"]) + " v " + str(c["away"]))[:42]
            price = ("  price %3.0f%%" % (100 * c["market_p"])
                     if c["market_p"] is not None else "  unpriced")
            extra = ("  double chance %3.0f%%" % (100 * c["p_double_chance"])
                     if dc and c["p_double_chance"] else "")
            print("   %-16s %-42s %-22s %3.0f%%%s%s"
                  % (ko, match, ("-> " + str(c["side"]))[:22], 100 * c["p"],
                     price, extra))

    show("BANKERS - short list: favourite 75%+, price agrees", out["bankers"],
         dc=True)
    acca = out["accumulator"]
    if acca:
        dc = acca["all_win_double_chance"]
        print("   all %d winning together: %.0f%%%s"
              % (acca["legs"], 100 * acca["all_win"],
                 "   (as double chances: %.0f%%)" % (100 * dc) if dc else ""))
    show("LONG LIST - favourite 65%+, price agrees", out["long_list"])
    show("UNPRICED - no price to check against, not yet benchmarked",
         out["unpriced"])
    show("AVOID - the model and the price back different sides", out["avoid"])
    for n in out["notes"]:
        print("\nnote: " + n)
    e = tips.EVIDENCE
    print("\nMeasured on %s." % e["source"])
    print("   75%%+ favourites won %.1f%%, but flat-stake return was %+.1f%%: "
          "hit rate is not profit." % (100 * e["bankers"]["hit"],
                                       100 * e["bankers"]["flat_return"]))
    print("   A week's top 4 all won in %.1f%% of weeks; top 8 in %.1f%%."
          % (100 * e["accumulator_all_won"]["4"],
             100 * e["accumulator_all_won"]["8"]))


def cmd_leagues(a):
    df = loader.load(a.data)
    g = df.groupby("Div").agg(matches=("Date", "size"), first=("Date", "min"),
                              last=("Date", "max"))
    print("%-6s%-24s%-14s%9s  %s" % ("Code", "League", "Country", "Matches", "Covered"))
    print("-" * 78)
    for d, r in g.iterrows():
        star = " *" if d in leagues.TOP_10 else ""
        print("%-6s%-24s%-14s%9d  %s to %s%s"
              % (d, leagues.name(d), leagues.country(d), r["matches"],
                 r["first"].date(), r["last"].date(), star))
    print("\n* one of the top 10 European leagues")


# -------------------------------------------------------------------- parser
def _parse_when(v):
    if not v:
        return datetime.now()
    for f in ("%Y-%m-%d", "%d/%m/%Y", "%d %b %Y"):
        try:
            return datetime.strptime(v, f)
        except ValueError:
            continue
    raise SystemExit("Could not read date %r, use YYYY-MM-DD" % v)


def _weights(a):
    return {"goals": 1.0, "sot": max(0.0, a.sot_weight)}


def _asof(a):
    return _parse_when(a.as_of) if getattr(a, "as_of", None) else None


def build_parser():
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="Football score and betting-market predictor built on "
                    "European league results.")
    p.add_argument("--data", default=DEFAULT_DATA, help="folder of football-data CSVs")
    p.add_argument("--xi", type=float, default=DEFAULT_XI,
                   help="time decay per day, 0 disables it")
    p.add_argument("--goal-shrink", type=float, default=DEFAULT_GOAL_SHRINK,
                   help="pull total goals toward the league mean, 1 disables it")
    p.add_argument("--edge-scale", type=float, default=DEFAULT_EDGE_SCALE,
                   help="pull the home/away split toward even, 1 disables it")
    p.add_argument("--sot-weight", type=float, default=DEFAULT_WEIGHTS["sot"],
                   help="weight on the shots-on-target model, 0 disables it")
    p.add_argument("--no-ladder", action="store_true",
                   help="do not carry promoted clubs' ratings up a division")
    p.add_argument("--market-weight", type=float, default=DEFAULT_MARKET_WEIGHT,
                   help="share of the forecast taken from the closing price "
                        "when one is available, 0 disables it")
    p.add_argument("--as-of", help="pretend today is this date, YYYY-MM-DD")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, top10=True):
        sp.add_argument("-l", "--league", help="division code, e.g. E0, SP1, I1")
        if top10:
            sp.add_argument("--top10", action="store_true",
                            help="restrict to the top 10 European leagues")

    s = sub.add_parser("predict", help="one match card")
    s.add_argument("match", nargs="+", help="two team names, or 'A vs B'")
    common(s, top10=False)
    s.add_argument("-d", "--date", help="match date shown on the card")
    s.add_argument("-f", "--full", action="store_true",
                   help="every market, not just the main card")
    s.add_argument("-b", "--best", action="store_true",
                   help="append the strongest selections")
    s.add_argument("-n", "--scores", type=int, default=5, help="correct scores to list")
    s.add_argument("--neutral", action="store_true", help="drop home advantage")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("slate", help="predict every upcoming fixture")
    common(s)
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--limit", type=int, default=12, help="fixtures per league")
    s.add_argument("--fixtures", help="fixtures CSV to use")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_slate)

    s = sub.add_parser("brief", help="write a date window of fixtures to PDF")
    common(s)
    s.add_argument("-o", "--out", help="output PDF path (default predictions.pdf)")
    s.add_argument("--start", help="first date, YYYY-MM-DD (default today)")
    s.add_argument("--end", help="last date, YYYY-MM-DD (default start + 2 days)")
    s.add_argument("--after", help="on the start date, only kick-offs at or "
                                   "after this time, e.g. 17:00")
    s.add_argument("--fixtures", help="fixtures CSV to use")
    s.add_argument("--tz", default="Africa/Dar_es_Salaam",
                   help="time zone for kick-off times (feed times are UK)")
    s.add_argument("--note", action="append",
                   help="a line for the coverage notes; repeatable")
    s.add_argument("--top-pairs", type=int, metavar="N",
                   help="price every pairing among each league's top N clubs, "
                        "for when no schedule is published")
    s.add_argument("--pair", action="append",
                   help="price a matchup with no schedule, e.g. "
                        "\"N1:Ajax v PSV Eindhoven\"; repeatable")
    s.add_argument("--dissent", type=int, default=25,
                   help="how many model/price disagreements to list")
    s.set_defaults(func=cmd_brief)

    s = sub.add_parser("fixtures", help="show the upcoming schedule")
    common(s)
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--fixtures", help="fixtures CSV to use")
    s.set_defaults(func=cmd_fixtures)

    s = sub.add_parser("refresh-fixtures", help="download the upcoming fixtures feed")
    s.set_defaults(func=cmd_refresh)

    s = sub.add_parser("refresh-results",
                       help="pull the current season's results into the data "
                            "folder, merging with what is already there")
    s.add_argument("--divs", nargs="*",
                   help="division codes to refresh (default: all on "
                        "football-data)")
    s.add_argument("--since", type=int, default=None,
                   help="also re-pull every season back to this start year, "
                        "e.g. 2023 rebuilds three seasons of history")
    s.set_defaults(func=cmd_refresh_results)

    s = sub.add_parser("update",
                       help="refresh everything online: football-data results "
                            "then the fixtures feed (additive only)")
    s.add_argument("--since", type=int, default=None,
                   help="also re-pull every season back to this start year")
    s.set_defaults(func=cmd_update)

    s = sub.add_parser("refresh-openfootball",
                       help="refresh African divisions from the public-domain "
                            "openfootball/world repo (second source)")
    s.add_argument("--leagues", nargs="*",
                   help="division codes to refresh (default: all covered)")
    s.add_argument("--raw", default=os.path.join(
        os.path.dirname(__file__), os.pardir, "data", "raw"),
                   help="where openfootball txt files are kept (default data/raw)")
    s.add_argument("--outdir", default=os.path.join(
        os.path.dirname(__file__), os.pardir, "data", "leagues"),
                   help="where rebuilt league CSVs go (default data/leagues)")
    s.set_defaults(func=cmd_refresh_openfootball)

    s = sub.add_parser("refresh-tanzania",
                       help="results and dated fixtures for the NBC Premier "
                            "League, from ligikuu.co.tz (third source)")
    s.add_argument("--url", default="https://ligikuu.co.tz/",
                   help="league homepage to read (default ligikuu.co.tz)")
    s.add_argument("--root", default=os.path.join(
        os.path.dirname(__file__), os.pardir),
                   help="project root holding data/manual (default: the repo)")
    s.add_argument("--raw", default=os.path.join(
        os.path.dirname(__file__), os.pardir, "data", "raw"),
                   help="where openfootball txt files are kept (default data/raw)")
    s.add_argument("--outdir", default=os.path.join(
        os.path.dirname(__file__), os.pardir, "data", "leagues"),
                   help="where rebuilt league CSVs go (default data/leagues)")
    s.set_defaults(func=cmd_refresh_tanzania)

    def record_args(sp):
        sp.add_argument("--repo", default=os.path.join(
            os.path.dirname(__file__), os.pardir),
            help="where data/record lives (default: the repo)")

    s = sub.add_parser("record-publish",
                       help="freeze the coming fixtures into the append-only "
                            "public record (only ones yet to kick off)")
    record_args(s)
    s.add_argument("--days", type=int, default=2,
                   help="how far ahead to publish (default 2)")
    s.set_defaults(func=cmd_record_publish)

    s = sub.add_parser("record",
                       help="the public record: predicted, then what happened")
    record_args(s)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("record-verify",
                       help="check the record's hash chain for tampering")
    record_args(s)
    s.set_defaults(func=cmd_record_verify)

    s = sub.add_parser("record-migrate",
                       help="copy the file record into DATABASE_URL, keeping "
                            "its hash chain (run once when switching)")
    record_args(s)
    s.set_defaults(func=cmd_record_migrate)

    s = sub.add_parser("refresh-fixtures-api",
                       help="fixtures for leagues and cups from API-Football "
                            "(one request per day fetched)")
    s.add_argument("--days", type=int, default=2,
                   help="today and how many days after (default 2)")
    s.add_argument("--root", default=os.path.join(
        os.path.dirname(__file__), os.pardir),
                   help="project root holding data/manual (default: the repo)")
    s.set_defaults(func=cmd_refresh_fixtures_api)

    s = sub.add_parser("tips",
                       help="clear picks: short list, long list, what to avoid")
    s.add_argument("--days", type=int, default=2,
                   help="how far ahead to look (default 2)")
    s.set_defaults(func=cmd_tips)

    s = sub.add_parser("live-leagues",
                       help="look up API-Football league ids "
                            "(spends one request of the live budget)")
    s.add_argument("query", nargs="+", help="search text, e.g. tanzania")
    s.set_defaults(func=cmd_live_leagues)

    s = sub.add_parser("live-status",
                       help="live-score budget and last call (spends nothing)")
    s.set_defaults(func=cmd_live_status)

    s = sub.add_parser("table", help="team attack and defence ratings")
    common(s)
    s.set_defaults(func=cmd_table)

    s = sub.add_parser("form", help="a team's recent results and ratings")
    s.add_argument("team", nargs="+")
    common(s, top10=False)
    s.add_argument("-n", type=int, default=6)
    s.set_defaults(func=cmd_form)

    s = sub.add_parser("backtest", help="walk-forward accuracy and calibration")
    common(s, top10=False)
    s.add_argument("--min-train", type=int, default=180)
    s.add_argument("--refit-days", type=int, default=7)
    s.add_argument("--edge", type=float, default=0.05)
    s.add_argument("--out", help="write per-match predictions to CSV")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("leagues", help="what data is loaded")
    s.set_defaults(func=cmd_leagues)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if not os.path.isdir(a.data):
        raise SystemExit("Data folder not found: %s" % a.data)
    a.func(a)


if __name__ == "__main__":
    main()
