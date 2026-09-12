"""League metadata for the football-data.co.uk division codes."""

# div code -> (display name, country, tier, is_top10_europe)
LEAGUES = {
    "E0":  ("Premier League",        "England",     1, True),
    "SP1": ("La Liga",               "Spain",       1, True),
    "I1":  ("Serie A",               "Italy",       1, True),
    "D1":  ("Bundesliga",            "Germany",     1, True),
    "F1":  ("Ligue 1",               "France",      1, True),
    "P1":  ("Primeira Liga",         "Portugal",    1, True),
    "N1":  ("Eredivisie",            "Netherlands", 1, True),
    "B1":  ("Jupiler Pro League",    "Belgium",     1, True),
    "T1":  ("Super Lig",             "Turkey",      1, True),
    "G1":  ("Super League Greece",   "Greece",      1, True),
    "SC0": ("Scottish Premiership",  "Scotland",    1, False),
    "E1":  ("Championship",          "England",     2, False),
    "E2":  ("League One",            "England",     3, False),
    "E3":  ("League Two",            "England",     4, False),
    "EC":  ("National League",       "England",     5, False),
    # Added so European ties involving these countries can be priced. Results
    # only - no shots and no prices - so they run on the goals model alone.
    "AUT1": ("Austrian Bundesliga",   "Austria",     1, False),
    "NOR1": ("Eliteserien",           "Norway",      1, False),
    "CZE1": ("Czech First League",    "Czechia",     1, False),
    "UKR1": ("Ukrainian Premier League", "Ukraine",  1, False),
    # Africa. Results only, from openfootball; used for CAF competitions and
    # for the home market.
    "TZ1": ("NBC Premier League",     "Tanzania",    1, False),
    "EG1": ("Egyptian Premier League", "Egypt",      1, False),
    "MA1": ("Botola Pro",             "Morocco",     1, False),
    "DZ1": ("Algerian Ligue 1",       "Algeria",     1, False),
    "ZA1": ("South African Premiership", "South Africa", 1, False),
    "NG1": ("Nigeria Premier League", "Nigeria",     1, False),
    "GH1": ("Ghana Premier League",   "Ghana",       1, False),
    "KE1": ("Kenyan Premier League",  "Kenya",       1, False),
    "UG1": ("Uganda Premier League",  "Uganda",      1, False),
    "ZM1": ("Zambia Super League",    "Zambia",      1, False),
    "RW1": ("Rwanda Premier League",  "Rwanda",      1, False),
}

TOP_10 = [c for c, m in LEAGUES.items() if m[3]]


def name(div):
    return LEAGUES.get(div, (div, "", 0, False))[0]


def country(div):
    return LEAGUES.get(div, (div, "", 0, False))[1]


def label(div):
    m = LEAGUES.get(div)
    return f"{m[0]} ({m[1]})" if m else div


# Divisions that promote and relegate into each other, strongest first. Only
# ladders where both rungs are actually loaded can transfer ratings.
LADDERS = [["E0", "E1", "E2", "E3", "EC"]]

# How much a club's rating shifts when it drops one rung, measured from clubs
# that actually moved: (attack, defence) in the LOWER division minus the same
# club in the UPPER one. From scratch/ladder.py, 48 movers across four steps.
RUNG_OFFSET = {
    ("E0", "E1"): (0.608, -0.633),
    ("E1", "E2"): (0.336, -0.420),
    ("E2", "E3"): (0.206, -0.228),
    ("E3", "EC"): (0.275, -0.171),
}


def ladder_of(div):
    """The promotion ladder containing `div`, or None."""
    for lad in LADDERS:
        if div in lad:
            return lad
    return None


def rating_shift(frm: str, to: str):
    """(attack, defence) to add to a rating measured in `frm` to express it
    in `to`. Returns None when the two divisions are not on one ladder."""
    lad = ladder_of(frm)
    if lad is None or to not in lad:
        return None
    i, j = lad.index(frm), lad.index(to)
    if i == j:
        return (0.0, 0.0)
    da = dd = 0.0
    step = 1 if j > i else -1
    for k in range(i, j, step):
        upper, lower = (lad[k], lad[k + 1]) if step > 0 else (lad[k - 1], lad[k])
        off = RUNG_OFFSET.get((upper, lower))
        if off is None:
            return None
        # moving down a rung adds the offset, moving up subtracts it
        da += off[0] * step
        dd += off[1] * step
    return (da, dd)


# Per-league shrinkage overrides. The defaults were tuned across the ten top
# European leagues; a league whose shape differs enough gets its own, measured
# the same way (walk-forward out-of-sample log-loss). See scratch/tune_tz.py.
LEAGUE_PARAMS = {
    # Tanzania is far more top-heavy than any European league, so its raw
    # ratings are the most over-dispersed on totals and need pulling in hard.
    "TZ1": {"goal_shrink": 0.25, "edge_scale": 1.10},
}


def params_for(div, goal_shrink, edge_scale):
    """League-specific shrinkage where it was measured, defaults otherwise."""
    o = LEAGUE_PARAMS.get(div, {})
    return o.get("goal_shrink", goal_shrink), o.get("edge_scale", edge_scale)
