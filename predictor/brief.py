"""Render a slate of upcoming fixtures as a PDF briefing."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import leagues

INK = colors.HexColor("#12181F")
MUTED = colors.HexColor("#5C6773")
LINE = colors.HexColor("#D4DCE4")
BAND = colors.HexColor("#EEF2F6")
ACCENT = colors.HexColor("#0E6F78")
WARN = colors.HexColor("#9A5B12")
GOOD = colors.HexColor("#1B7F5A")


def _styles():
    ss = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=22, leading=25, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica",
                              fontSize=10, leading=14, textColor=MUTED),
        "h2": ParagraphStyle("h2", parent=ss["Normal"], fontName="Helvetica-Bold",
                             fontSize=12, leading=15, textColor=INK,
                             spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=8.6, leading=12, textColor=INK),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontName="Helvetica",
                                fontSize=7.6, leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=7.8, leading=9.6, textColor=INK),
    }
    return s


def _table(data, widths, align_right=(), highlight_col=None):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.2),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 7.8),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, LINE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#E8EDF2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    if highlight_col is not None:
        style += [("FONT", (highlight_col, 1), (highlight_col, -1), "Helvetica-Bold", 7.8),
                  ("TEXTCOLOR", (highlight_col, 1), (highlight_col, -1), ACCENT)]
    t.setStyle(TableStyle(style))
    return t


def _pct(x):
    return "%d%%" % round(100 * x)


def _order_any(r):
    d = r.get("date")
    return (d if hasattr(d, "strftime") else pd.Timestamp.max, r.get("time", ""))


def _when(r):
    """'Sat 12 Sep' from a row's date, which may be a timestamp or a note."""
    d = r.get("date")
    if hasattr(d, "strftime"):
        return d.strftime("%a %d %b")
    return str(d) if d is not None else ""


def build(rows, out_path: str, window_label: str, generated: datetime,
          meta: dict, dissent=None, skipped=None, promoted=None,
          cross=None, caf=None) -> str:
    """Write the briefing. `rows` are dicts produced by cmd_brief."""
    st = _styles()
    tz = meta.get("tz_label", "UK")
    caf = caf or []
    doc = SimpleDocTemplate(
        out_path, pagesize=landscape(A4),
        leftMargin=13 * mm, rightMargin=13 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
        title="Football predictions - %s" % window_label,
        author="football-predictor")

    story = []
    story.append(Paragraph("Football predictions", st["title"]))
    story.append(Paragraph(
        "%s &nbsp;|&nbsp; %d fixtures &nbsp;|&nbsp; all kick-off times in %s "
        "&nbsp;|&nbsp; generated %s %s"
        % (window_label, len(rows) + len(caf), tz,
           generated.strftime("%a %d %b %Y, %H:%M"), tz),
        st["sub"]))
    story.append(Spacer(1, 7))

    # ---- what produced these numbers, and what they are not ---------------
    story.append(Paragraph(
        "<b>How these were produced.</b> A Dixon-Coles bivariate Poisson model "
        "fitted per league on %s matches of results, weighted so recent games "
        "count more, blended with a second model fitted on shots on target. "
        "Where a closing price exists it is de-vigged and folded in at %d%% "
        "weight. Every market below is read off one scoreline distribution per "
        "fixture, so no two numbers on a row can contradict each other."
        % (meta.get("n_matches", "14,788"), round(100 * meta.get("market_weight", 0))),
        st["body"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>What this is not.</b> Over 5,948 walk-forward tests this engine "
        "scores a log-loss of 0.9568 against the bookmakers' closing consensus "
        "at 0.9558 - it matches the market, it does not beat it. Betting "
        "the model's disagreements with the price lost 12.2% of stakes over "
        "3,761 historical bets. Treat these as calibrated estimates of how "
        "likely things are, not as tips with an edge.",
        st["body"]))
    story.append(Spacer(1, 9))

    # ---- CAF Champions League, first -----------------------------------------
    if caf:
        story.append(Paragraph(
            "CAF Champions League &nbsp;<font color='#5C6773'>| first "
            "preliminary round, return legs | %d</font>" % len(caf), st["h2"]))
        story.append(Paragraph(
            "Priced through an African country bridge measured from 199 CAF "
            "club matches (2023/24-2024/25), which beat having no bridge by "
            "5.8% out of sample. CAF home advantage measures +0.59 - nearly "
            "double Europe's - so the return leg at home is worth a great deal. "
            "A club from a league loaded here is rated from its own results; a "
            "club from a federation with no loaded league is rated as an "
            "<b>average club of that federation</b>, from its CAF record alone. "
            "No prices exist for these, so every number is the model unaided.",
            st["body"]))
        story.append(Spacer(1, 5))
        ch = ["Date", "Time (%s)" % tz, "Match", "1", "X", "2", "Score", "xG",
              "Home rated from", "Away rated from"]
        cw = [19 * mm, 17 * mm, 58 * mm, 11 * mm, 11 * mm, 11 * mm, 13 * mm,
              22 * mm, 55 * mm, 55 * mm]
        cd = [ch]
        for r in sorted(caf, key=_order_any):
            cd.append([_when(r), r["time"], r["match"] + (" *" if r["thin"] else ""),
                       _pct(r["pH"]), _pct(r["pD"]), _pct(r["pA"]), r["score"],
                       "%.2f - %.2f" % (r["xgh"], r["xga"]),
                       r["home_basis"], r["away_basis"]])
        story.append(_table(cd, cw, align_right=(3, 4, 5)))
        story.append(Spacer(1, 6))

        ties = [r for r in caf if r.get("tie")]
        if ties:
            story.append(Paragraph(
                "<b>Who goes through.</b> The first-leg score is carried into the "
                "second leg's scoreline distribution. Level on aggregate after 90 "
                "minutes is shown on its own rather than resolved, because it "
                "goes to extra time and penalties, which this engine does not "
                "model.", st["body"]))
            story.append(Spacer(1, 4))
            th = ["Tie (second leg)", "First leg", "Through in 90 min",
                  "Level - extra time / pens", "Out in 90 min"]
            td = [th]
            for r in ties:
                h1, a1 = r["leg1"]
                td.append([r["match"],
                           "%s scored %d, %s scored %d" % (
                               r["match"].split(" v ")[0], h1,
                               r["match"].split(" v ")[1], a1),
                           _pct(r["tie"]["win"]), _pct(r["tie"]["level"]),
                           _pct(r["tie"]["lose"])])
            story.append(_table(td, [70 * mm, 88 * mm, 32 * mm, 42 * mm, 28 * mm],
                                align_right=(2, 3, 4)))
            story.append(Spacer(1, 5))
        story.append(Paragraph(
            "* No CAF record at all for that federation in the data, so it is "
            "rated as an average CAF entrant - which almost certainly flatters "
            "it. Blind test before publishing: on the first legs, played after "
            "all of this data, the model gave Simba 65% to win in Lilongwe (they "
            "won 0-1, the likeliest single score) and Young Africans 88% in "
            "Gaborone (it finished 1-1, a one-in-ten outcome). Extreme home "
            "numbers for Simba and Young Africans are consistent with their own "
            "CAF home record against small federations - 5-1, 6-0 and 6-0 - but "
            "no football match is really 99% certain.", st["small"]))
        story.append(Spacer(1, 10))

    # ---- the slate, league by league ---------------------------------------
    header = ["Date", "Time (%s)" % tz, "Match", "1", "X", "2", "Pick",
              "O2.5", "U2.5", "BTTS", "Score", "xG", "Confidence"]
    widths = [19 * mm, 17 * mm, 60 * mm, 11 * mm, 11 * mm, 11 * mm, 13 * mm,
              12 * mm, 12 * mm, 12 * mm, 14 * mm, 22 * mm, 25 * mm]

    by_league = {}
    for r in rows:
        by_league.setdefault(r["league"], []).append(r)

    def rank(name):
        for i, code in enumerate(leagues.TOP_10):
            if leagues.label(code) == name:
                return (0, i)
        return (1, name)

    def _order(r):
        d = r.get("date")
        return (d if hasattr(d, "strftime") else pd.Timestamp.max, r["time"])

    for league in sorted(by_league, key=rank):
        items = sorted(by_league[league], key=_order)
        data = [header]
        for r in items:
            data.append([
                _when(r), r["time"], r["match"],
                _pct(r["pH"]), _pct(r["pD"]), _pct(r["pA"]),
                r["pick"], _pct(r["over25"]), _pct(1 - r["over25"]),
                _pct(r["btts"]), r["score"],
                "%.2f - %.2f" % (r["xgh"], r["xga"]), r["confidence"],
            ])
        block = [Paragraph("%s &nbsp;<font color='#5C6773'>| %d</font>"
                           % (league, len(items)), st["h2"]),
                 _table(data, widths, align_right=(3, 4, 5, 7, 8, 9),
                        highlight_col=6)]
        # Keep a league's heading on the same page as its table; only a table
        # too long for one page is allowed to split.
        story.append(KeepTogether(block) if len(items) <= 28 else block[0])
        if len(items) > 28:
            story.append(block[1])
        story.append(Spacer(1, 5))

    # ---- where the model disagrees with the price --------------------------
    if dissent:
        story.append(PageBreak())
        story.append(Paragraph("Where the model disagrees with the price", st["title"]))
        story.append(Spacer(1, 5))
        story.append(Paragraph(
            "These are the fixtures where the model's own view - before "
            "any price was folded in - differs most from the bookmakers. "
            "They are shown because a disagreement is the interesting part of a "
            "forecast, not because they are good bets. Historically this exact "
            "filter lost money. The model column is the pure model; the price "
            "column is the de-vigged closing consensus.", st["body"]))
        story.append(Spacer(1, 8))
        dh = ["Match", "League", "Selection", "Model", "Price implies",
              "Difference", "Decimal odds"]
        dw = [66 * mm, 40 * mm, 26 * mm, 20 * mm, 26 * mm, 24 * mm, 24 * mm]
        dd = [dh]
        for r in dissent:
            dd.append([r["match"], r["league"], r["sel"], _pct(r["model_p"]),
                       _pct(r["market_p"]), "%+.0f pts" % (100 * r["diff"]),
                       "%.2f" % r["price"]])
        story.append(_table(dd, dw, align_right=(3, 4, 5, 6)))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "A large difference usually means the model is missing something the "
            "market knows: an injury, a suspension, a rotated side, a manager "
            "change. It reads only results and shots.", st["small"]))

    # ---- caveats -----------------------------------------------------------
    story.append(Spacer(1, 10))
    notes = []
    if promoted:
        notes.append("<b>Clubs with no history in their division.</b> "
                     + "; ".join(sorted(set(promoted))) + ".")
    if skipped:
        notes.append("<b>Not predicted.</b> " + "; ".join(skipped) + ".")
    if cross:
        story.append(PageBreak())
        story.append(Paragraph("Cross-league ties - listed, not priced", st["h2"]))
        story.append(Paragraph(
            "These pair clubs from different domestic leagues where <b>no "
            "measured bridge covers both sides</b>, so no probabilities are "
            "given. Ties between leagues the bridges do cover are priced in "
            "the tables above; these are the ones left over, and publishing a "
            "number for them would be a guess wearing the same clothes as the "
            "measured forecasts.", st["body"]))
        story.append(Spacer(1, 5))
        story.append(Paragraph(
            "What follows is each club's standing <b>within its own league</b>, "
            "where 1.00 is that league's average side. Real and useful, but do "
            "not read across a row as a head-to-head.", st["body"]))
        story.append(Spacer(1, 7))
        cdata = [["Date", "Match", "Home league", "Rating",
                  "Away league", "Rating"]]
        for c in cross:
            h, a = c["sides"]
            when = (c["date"].strftime("%a %d %b") if c.get("date") is not None
                    and hasattr(c["date"], "strftime") else "")
            cdata.append([("%s %s" % (when, c["time"])).strip(),
                          "%s v %s" % (h["team"], a["team"]),
                          h["league"], "%.2f" % h["rating"],
                          a["league"], "%.2f" % a["rating"]])
        story.append(_table(cdata, [26 * mm, 66 * mm, 46 * mm, 18 * mm,
                                    46 * mm, 18 * mm]))
        story.append(Spacer(1, 9))

    notes.append(
        "<b>Reading the numbers.</b> 1 / X / 2 are home win, draw and away win. "
        "O2.5 and U2.5 are over and under 2.5 total goals. BTTS is both teams "
        "to score. Score is the single most likely scoreline, which is often "
        "not the same as the most likely result: a favourite's win is "
        "spread across many scorelines while draws concentrate on 1-1.")
    notes.append(
        "<b>Data.</b> Results and prices from football-data.co.uk, free for "
        "personal use. Austrian, Norwegian, Czech and African results, and CAF "
        "and UEFA club competition results, from openfootball (public domain); "
        "current Tanzanian results and fixtures from the league's own site, "
        "ligikuu.co.tz. Kick-off times are converted to %s from the zone each "
        "source publishes in - UK for the football-data feed, East Africa for "
        "the Tanzanian league site." % tz)
    extra = meta.get("notes") or []
    if extra:
        story.append(Spacer(1, 4))
        # Not "this weekend": briefs are built for any window, and a Tuesday
        # brief headed with the weekend reads as a stale template.
        story.append(Paragraph("Coverage and caveats", st["h2"]))
        for n in extra:
            story.append(Paragraph(n, st["body"]))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 6))
    for n in notes:
        story.append(Paragraph(n, st["small"]))
        story.append(Spacer(1, 3))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(13 * mm, 7 * mm,
                          "Estimates from historical results and prices. "
                          "No edge over the market is claimed or implied. "
                          "18+. Gamble responsibly.")
        canvas.drawRightString(landscape(A4)[0] - 13 * mm, 7 * mm,
                               "Page %d" % doc_.page)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path
