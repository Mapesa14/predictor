"""European club competition results, used to make leagues comparable.

Each domestic league is fitted with its own zero-sum constraint, so an English
attack rating and an Italian one are not the same quantity. Matches between
clubs of different countries are the only thing that ties the scales together,
exactly as clubs moving divisions tie the promotion ladder together.

Source: the openfootball/champions-league repository (public domain), parsed
from its football.txt format.
"""
from __future__ import annotations

import glob
import os
import re

import pandas as pd

# UEFA three-letter country code -> the division we load for that country
COUNTRY_DIV = {
    "ENG": "E0", "ESP": "SP1", "ITA": "I1", "GER": "D1", "FRA": "F1",
    "NED": "N1", "POR": "P1", "BEL": "B1", "TUR": "T1", "GRE": "G1",
    "SCO": "SC0", "AUT": "AUT1", "NOR": "NOR1", "CZE": "CZE1", "UKR": "UKR1",
}

_DATE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})"
    r"(?:\s+(\d{4}))?\s*$")

# "  18:45  Home (ENG)  v Away (ESP)   2-1 (1-0)"   time optional
_MATCH = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(.+?)\s+\((?P<hc>[A-Z]{3})\)\s+v\s+(.+?)\s+\((?P<ac>[A-Z]{3})\)"
    r"\s+(?P<hg>\d+)\s*-\s*(?P<ag>\d+)")

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"], 1)}

# Corporate noise that never appears in football-data's short names.
_NOISE = re.compile(
    r"\b(FC|CF|ACF|AC|SSC|SC|AS|SS|SK|SV|SL|CD|RCD|RC|BSC|BC|VfB|VfL|FK|KV|"
    r"KAA|KRC|RSC|NK|GNK|HNK|PFC|PAE|SFP|OGC|OSC|CFR|MOL|TSG|SpVgg|JK|"
    r"Olympique|Calcio|Balompie|Praia|1899|1900|1902|1904|1907|1909|1913|"
    r"1846|1848|04|05|07|96)\b", re.I)

# City words appended to a club name that football-data leaves off.
_CITY = re.compile(
    r"\s+(Piraeus|Saloniki|Thessaloniki|Athen|Athens|Rotterdam|Milano|"
    r"Munchen|Munich|Enschede|Arnhem|Alkmaar|Praha|Prague|Zagreb|Beograd|"
    r"Lisboa|Torino|Bergamo|Firenze)$", re.I)


def _clean(name: str) -> str:
    """Strip corporate tokens and accents so names can be matched."""
    s = name.strip()
    s = (s.replace("ü", "u").replace("ö", "o").replace("ä", "a")
          .replace("é", "e").replace("è", "e").replace("á", "a")
          .replace("í", "i").replace("ó", "o").replace("ú", "u")
          .replace("ç", "c").replace("ñ", "n").replace("Š", "S")
          .replace("š", "s").replace("ø", "o").replace("å", "a")
          .replace("ã", "a").replace("â", "a").replace("ł", "l")
          .replace("ș", "s").replace("ț", "t").replace("ı", "i")
          .replace("ş", "s").replace("Ş", "S").replace("ğ", "g")
          .replace("Ğ", "G").replace("İ", "I").replace("ć", "c")
          .replace("č", "c").replace("ž", "z").replace("Ž", "Z")
          .replace("ö", "o").replace("Ö", "O").replace("Ü", "U"))
    s = re.sub(r"^\s*\d+\.\s*", "", s)       # leading "1. " in German names
    s = _NOISE.sub(" ", s)
    s = " ".join(s.split()).strip(" .-")
    return s


def _drop_city(s: str) -> str:
    """A second attempt with any trailing city name removed."""
    out = _CITY.sub("", s).strip()
    return out if out and out != s else ""


_SEASON = re.compile(r"(\d{4})-(\d{2})")


def _season_years(path: str):
    """(first year, second year) from a filename like 2024-25_cl.txt."""
    m = _SEASON.search(os.path.basename(path))
    if not m:
        return None, None
    y1 = int(m.group(1))
    return y1, int(str(y1)[:2] + m.group(2))


def parse_file(path: str) -> list[dict]:
    """Read one football.txt file into match dicts."""
    y1, y2 = _season_years(path)
    out, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith(("=", "#")):
                continue
            d = _DATE.match(line)
            if d:
                mon, day, yr = d.group(1), int(d.group(2)), d.group(3)
                month = _MONTHS[mon]
                # A season spans two calendar years; the month says which.
                year = int(yr) if yr else (
                    y1 if (y1 and month >= 7) else y2)
                cur = (pd.Timestamp(year=year, month=month, day=day)
                       if year else None)
                continue
            m = _MATCH.match(line)
            if m and cur is not None:
                out.append({
                    "Date": cur,
                    "home_raw": m.group(1).strip(), "hc": m.group("hc"),
                    "away_raw": m.group(3).strip(), "ac": m.group("ac"),
                    "FTHG": int(m.group("hg")), "FTAG": int(m.group("ag")),
                    "source": os.path.basename(path),
                })
    return out


def load(root: str) -> pd.DataFrame:
    """Every European tie found under `root`."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*.txt"))):
        rows += parse_file(p)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["home_div"] = df["hc"].map(COUNTRY_DIV)
    df["away_div"] = df["ac"].map(COUNTRY_DIV)
    df["home_clean"] = df["home_raw"].map(_clean)
    df["away_clean"] = df["away_raw"].map(_clean)
    return df.drop_duplicates(
        subset=["Date", "home_raw", "away_raw"]).sort_values("Date")


# Clubs whose official name shares too little with football-data's short form.
ALIASES = {
    "Borussia Dortmund": "Dortmund",
    "Dortmund": "Dortmund",
    "Club Atletico de Madrid": "Ath Madrid",
    "Atletico de Madrid": "Ath Madrid",
    "Real Sociedad de Futbol": "Sociedad",
    "Racing Club de Lens": "Lens",
    "Union Berlin": "Union Berlin",
    "Lazio Roma": "Lazio",
    "Sporting CP": "Sp Lisbon",
    "Sporting": "Sp Lisbon",
    "Heidenheim": "Heidenheim",
    "Istanbul Basaksehir": "Buyuksehyr",
    "Internazionale Milano": "Inter",
    "Milan": "Milan",
    "Bayern Munchen": "Bayern Munich",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Bayer Leverkusen": "Leverkusen",
    "Sport Lisboa e Benfica": "Benfica",
    "Sporting Clube de Portugal": "Sp Lisbon",
    "Sporting Braga": "Sp Braga",
    "Sporting Clube de Braga": "Sp Braga",
    "Atletico de Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Athletic Club": "Ath Bilbao",
    "Paris Saint-Germain": "Paris SG",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Borussia Monchengladbach": "M'gladbach",
    "Monchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Union Berlin": "Union Berlin",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
    "Werder Bremen": "Werder Bremen",
    "TSG Hoffenheim": "Hoffenheim",
    "Hoffenheim": "Hoffenheim",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "Marseille": "Marseille",
    "Stade Rennais": "Rennes",
    "Lille OSC": "Lille",
    "Lille": "Lille",
    "Stade Brestois 29": "Brest",
    "Brestois": "Brest",
    "Monaco": "Monaco",
    "Nice": "Nice",
    "Lens": "Lens",
    "Toulouse": "Toulouse",
    "PSV": "PSV Eindhoven",
    "PSV Eindhoven": "PSV Eindhoven",
    "Feyenoord Rotterdam": "Feyenoord",
    "AFC Ajax": "Ajax",
    "Ajax": "Ajax",
    "AZ Alkmaar": "AZ Alkmaar",
    "AZ": "AZ Alkmaar",
    "FC Twente": "Twente",
    "Twente Enschede": "Twente",
    "Utrecht": "Utrecht",
    "Vitesse Arnhem": "Vitesse",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "Club Brugge KV": "Club Brugge",
    "Club Brugge": "Club Brugge",
    "Royale Union Saint-Gilloise": "St. Gilloise",
    "Union Saint-Gilloise": "St. Gilloise",
    "RSC Anderlecht": "Anderlecht",
    "Anderlecht": "Anderlecht",
    "KAA Gent": "Gent",
    "Gent": "Gent",
    "Royal Antwerp": "Antwerp",
    "Antwerp": "Antwerp",
    "Cercle Brugge": "Cercle Brugge",
    "Galatasaray": "Galatasaray",
    "Fenerbahce": "Fenerbahce",
    "Besiktas JK": "Besiktas",
    "Besiktas": "Besiktas",
    "Trabzonspor": "Trabzonspor",
    "Istanbul Basaksehir": "Buyuksehyr",
    "Basaksehir": "Buyuksehyr",
    "Sivasspor": "Sivasspor",
    "Olympiacos Piraeus": "Olympiakos",
    "Olympiacos": "Olympiakos",
    "Olympiakos": "Olympiakos",
    "PAOK Thessaloniki": "PAOK",
    "PAOK": "PAOK",
    "AEK Athens": "AEK",
    "Panathinaikos": "Panathinaikos",
    "Aris Thessaloniki": "Aris",
    "Celtic": "Celtic",
    "Rangers": "Rangers",
    "Heart of Midlothian": "Hearts",
    "Hibernian": "Hibernian",
    "Real Betis Balompie": "Betis",
    "Real Betis": "Betis",
    "Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Sociedad": "Sociedad",
    "Villarreal": "Villarreal",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Girona": "Girona",
    "Celta de Vigo": "Celta",
    "Celta Vigo": "Celta",
    "Osasuna": "Osasuna",
    "Rayo Vallecano": "Vallecano",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Juventus": "Juventus",
    "Atalanta": "Atalanta",
    "Bologna": "Bologna",
    "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Udinese": "Udinese",
    "Como": "Como",
    "Porto": "Porto",
    "Vitoria Guimaraes": "Guimaraes",
    "Guimaraes": "Guimaraes",
    "Santa Clara": "Santa Clara",
    "Arouca": "Arouca",
    "Moreirense": "Moreirense",
    "Famalicao": "Famalicao",
    "Estoril Praia": "Estoril",
    "Rio Ave": "Rio Ave",
    "Gil Vicente": "Gil Vicente",
    "Casa Pia": "Casa Pia",
}


def resolve(df: pd.DataFrame, predictor) -> pd.DataFrame:
    """Attach football-data club names, dropping ties we cannot match."""
    if df.empty:
        return df
    out = df.copy()
    for side in ("home", "away"):
        names, ok = [], []
        for raw, clean, div in zip(out[side + "_raw"], out[side + "_clean"],
                                   out[side + "_div"]):
            name = _match_one(raw, clean, div, predictor)
            names.append(name)
            ok.append(name is not None)
        out[side] = names
        out[side + "_ok"] = ok
    return out


def _match_one(raw, clean, div, predictor):
    if not isinstance(div, str) or div not in predictor.divs:
        return None
    nocity = _drop_city(clean)
    for candidate in (ALIASES.get(clean), ALIASES.get(raw), ALIASES.get(nocity),
                      clean, nocity, raw):
        if not candidate:
            continue
        try:
            return predictor.resolve(candidate, div, fuzzy=False)[0]
        except SystemExit:
            continue
    return None


# --------------------------------------------------------------- the bridge
def build_dataset(euro_df, predictor, min_train=200, refit_days=7):
    """Attach as-of domestic ratings to each European tie.

    Ratings are taken from a model fitted only on matches played before the
    tie, so the offsets are calibrated the same honest way everything else in
    this project is measured.
    """
    import numpy as np
    from . import model as M

    d = euro_df.sort_values("Date").reset_index(drop=True)
    cache, rows = {}, []
    for _, x in d.iterrows():
        rec = {"Date": x.Date, "FTHG": x.FTHG, "FTAG": x.FTAG,
               "home": x.home, "away": x.away,
               "home_div": x.home_div, "away_div": x.away_div}
        ok = True
        for side, div, team in (("h", x.home_div, x.home),
                                ("a", x.away_div, x.away)):
            key = (div, x.Date.to_period("W").start_time)
            if key not in cache:
                try:
                    cache[key] = M.fit(predictor.df, div, "FT",
                                       xi=predictor.xi, as_of=x.Date,
                                       goal_shrink=1.0, edge_scale=1.0)
                except Exception:
                    cache[key] = None
            m = cache[key]
            if m is None or m.n_matches < min_train or not m.knows(team):
                ok = False
                break
            rec[side + "_att"] = m.attack[team]
            rec[side + "_def"] = m.defence[team] - m.mean_defence
            rec[side + "_base"] = m.base
        if ok:
            rows.append(rec)
    return pd.DataFrame(rows)


def fit_offsets(ds, divs=None, ridge=0.01, fit_beta=False):
    """Estimate one strength offset per league from cross-league results.

    log(home rate) = c + mean base + beta*(att_h + def_a) + s_home - s_away + adv
    log(away rate) = c + mean base + beta*(att_a + def_h) + s_away - s_home

    A club from a stronger league both scores more and concedes less against
    one from a weaker league, which is what the single offset expresses.

    beta is how much of a domestic rating survives the trip abroad. It is 1
    unless fitted: a club that dominates a weak league looks stronger at home
    than it is against another country's champion, so in a top-heavy league
    only part of the rating should transfer. The offsets are held to zero sum.
    """
    import numpy as np
    from scipy.optimize import minimize

    divs = divs or sorted(set(ds.home_div) | set(ds.away_div))
    idx = {d: i for i, d in enumerate(divs)}
    n = len(divs)
    hi = ds.home_div.map(idx).to_numpy(int)
    ai = ds.away_div.map(idx).to_numpy(int)
    hg = ds.FTHG.to_numpy(float)
    ag = ds.FTAG.to_numpy(float)
    lvl = 0.5 * (ds.h_base.to_numpy(float) + ds.a_base.to_numpy(float))
    h_att, h_def = ds.h_att.to_numpy(float), ds.h_def.to_numpy(float)
    a_att, a_def = ds.a_att.to_numpy(float), ds.a_def.to_numpy(float)

    def unpack(p):
        s = np.empty(n)
        s[:-1] = p[:n - 1]
        s[-1] = -s[:-1].sum()
        beta = p[n + 1] if fit_beta else 1.0
        return s, p[n - 1], p[n], beta

    def nll(p):
        s, c, adv, beta = unpack(p)
        lam = np.exp(c + lvl + beta * (h_att + a_def) + s[hi] - s[ai] + adv)
        mu = np.exp(c + lvl + beta * (a_att + h_def) + s[ai] - s[hi])
        lam, mu = np.clip(lam, 1e-6, 25), np.clip(mu, 1e-6, 25)
        ll = (hg * np.log(lam) - lam) + (ag * np.log(mu) - mu)
        return -float(ll.sum()) + ridge * len(ds) * float(np.sum(s ** 2))

    p0 = np.zeros(n + 1 + (1 if fit_beta else 0))
    p0[n] = 0.25
    bounds = [(-1.5, 1.5)] * (n - 1) + [(-2, 2), (-1, 1)]
    if fit_beta:
        p0[n + 1] = 1.0
        bounds.append((0.1, 1.5))
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 3000})
    s, c, adv, beta = unpack(res.x)
    return {"offsets": {d: float(s[i]) for d, i in idx.items()},
            "intercept": float(c), "home_adv": float(adv),
            "beta": float(beta), "n": len(ds)}


# ------------------------------------------------------------ CAF / Africa
CAF_COUNTRY_DIV = {
    "TAN": "TZ1", "EGY": "EG1", "MAR": "MA1", "ALG": "DZ1", "RSA": "ZA1",
    "NGA": "NG1", "GHA": "GH1", "KEN": "KE1", "UGA": "UG1", "ZAM": "ZM1",
    "RWA": "RW1",
}


def load_caf(root: str) -> pd.DataFrame:
    """CAF club competition results, with every federation kept.

    Unlike the UEFA set, most federations in a CAF draw have no league we can
    load. They are kept as federation-level entries rather than dropped, since
    a Malawian or Botswanan club's CAF record is the only evidence available
    about how strong that football is.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*.txt"))):
        rows += parse_file(p)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["Date", "home_raw", "away_raw"]).sort_values("Date")
    df["home_div"] = df["hc"].map(CAF_COUNTRY_DIV)
    df["away_div"] = df["ac"].map(CAF_COUNTRY_DIV)
    df["home_clean"] = df["home_raw"].map(_clean)
    df["away_clean"] = df["away_raw"].map(_clean)
    return df


def build_dataset_caf(caf_df, predictor, min_train=120):
    """As-of ratings for CAF ties; unloaded federations enter as 'F-XXX'.

    A club from a loaded league carries its own as-of rating. A club from a
    federation with no loaded league is treated as an average club of that
    federation, and the federation's offset carries all of its strength.
    """
    import numpy as np
    from . import model as M

    loaded = {d for d in CAF_COUNTRY_DIV.values() if d in predictor.divs}
    lvls = []
    for d in loaded:
        try:
            lvls.append(M.fit(predictor.df, d, "FT", xi=predictor.xi).base)
        except Exception:
            pass
    fed_base = float(np.mean(lvls)) if lvls else 0.0

    cache, rows = {}, []
    for _, x in caf_df.sort_values("Date").iterrows():
        rec = {"Date": x.Date, "FTHG": x.FTHG, "FTAG": x.FTAG}
        ok = True
        for side, code, raw, clean in (("h", x.hc, x.home_raw, x.home_clean),
                                       ("a", x.ac, x.away_raw, x.away_clean)):
            div = CAF_COUNTRY_DIV.get(code)
            team = _match_one(raw, clean, div, predictor) if div in loaded else None
            if div in loaded and team:
                key = (div, x.Date.to_period("W").start_time)
                if key not in cache:
                    try:
                        cache[key] = M.fit(predictor.df, div, "FT",
                                           xi=predictor.xi, as_of=x.Date,
                                           goal_shrink=1.0, edge_scale=1.0)
                    except Exception:
                        cache[key] = None
                m = cache[key]
                if m is None or m.n_matches < min_train or not m.knows(team):
                    ok = False
                    break
                rec[side + "_div"] = div
                rec[side + "_team"] = team
                rec[side + "_att"] = m.attack[team]
                rec[side + "_def"] = m.defence[team] - m.mean_defence
                rec[side + "_base"] = m.base
            else:
                # no league for this federation (or an unresolved name in one)
                rec[side + "_div"] = "F-" + code if div not in loaded else "F-" + code
                rec[side + "_team"] = raw
                rec[side + "_att"] = 0.0
                rec[side + "_def"] = 0.0
                rec[side + "_base"] = fed_base
        if ok and rec["h_div"] != rec["a_div"]:
            rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.rename(columns={"h_div": "home_div", "a_div": "away_div"})
    return out, fed_base
