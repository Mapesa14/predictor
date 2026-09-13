"""The predictor: data in, fitted models cached, match cards out."""
from __future__ import annotations

import difflib
import glob
import math
import os
from datetime import datetime

import pandas as pd

from . import fixtures, leagues, loader, market, markets, model

# All three tuned on walk-forward out-of-sample log-loss; see README and scratch/.
DEFAULT_XI = 0.0018           # time decay per day
DEFAULT_GOAL_SHRINK = 0.40    # pull the goal level toward the league mean
DEFAULT_EDGE_SCALE = 1.10     # sharpen the home/away split
# Goals say what happened; shots on target say the same thing with less noise.
# Equal weight was the out-of-sample optimum - see scratch/tune_shots.py.
DEFAULT_WEIGHTS = {"goals": 1.0, "sot": 1.0}
# Share of the published forecast taken from the closing price, when one is
# available. Tuned in scratch/tune_prior.py.
DEFAULT_MARKET_WEIGHT = 0.9


class Predictor:
    def __init__(self, root: str, divs: list[str] | None = None,
                 xi: float = DEFAULT_XI, as_of: datetime | None = None,
                 goal_shrink: float = DEFAULT_GOAL_SHRINK,
                 edge_scale: float = DEFAULT_EDGE_SCALE,
                 use_ladder: bool = True, weights: dict | None = None,
                 market_weight: float = DEFAULT_MARKET_WEIGHT):
        self.root = root
        self.xi = xi
        self.as_of = as_of
        self.goal_shrink = goal_shrink
        self.edge_scale = edge_scale
        self.use_ladder = use_ladder
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.bridge = load_bridge()
        self.caf_bridge = load_caf_bridge()
        self.market_weight = float(market_weight)
        self.df = loader.load(root, divs)
        self._models: dict[str, dict[str, model.GoalModel]] = {}

    # ------------------------------------------------------------- plumbing
    # `self.df` is set once and never mutated, so everything derived from it is
    # settled and worth holding on to. These caches are filled on first use
    # rather than in `__init__` because a Predictor is also built by `__new__`
    # in the tests, and because laziness and eagerness are identical for a
    # frame that cannot change. Each one exists for a measured reason: a slate
    # asks for them once per division and twice per fixture, and recomputing
    # them came to more than fitting all thirty models did.
    def _memo(self, name: str, make):
        got = self.__dict__.get(name)
        if got is None:
            got = self.__dict__[name] = make()
        return got

    @property
    def divs(self) -> list[str]:
        return self._memo("_divs", lambda: sorted(
            set(self.df["Div"].unique()),
            key=lambda d: (d not in leagues.TOP_10, d)))

    def models(self, div: str) -> dict[str, model.GoalModel]:
        if div not in self._models:
            gs, es = leagues.params_for(div, self.goal_shrink, self.edge_scale)
            m = model.fit_all(self.df, div, xi=self.xi, as_of=self.as_of,
                              goal_shrink=gs, edge_scale=es,
                              weights=self.weights)
            if "FT" not in m:
                raise SystemExit("Not enough data to fit %s" % div)
            if self.use_ladder:
                for gm in m.values():
                    self._carry_ratings_up(div, gm)
            self._models[div] = m
        return self._models[div]

    def clubs(self, div: str) -> list[dict]:
        """Rating table for one division.

        Each club's attack and defence are read the same way a fixture is
        (own record, carried rating, or the promoted prior), then expressed as
        expected goals against a league-average opponent. `str` is the logged
        strength index (attack minus defence), higher is better.
        """
        fm = self.models(div)["FT"]
        ha, base = fm.home_adv, fm.base
        rows = []
        for t in fm.teams:
            atk, dfs = fm._team(t)
            rows.append({
                "team": t,
                "attack": round(atk, 3),
                "defence": round(dfs, 3),
                "str": round(atk - dfs, 3),
                "gf_home": round(math.exp(ha + atk + base), 2),
                "ga": round(math.exp(dfs + base), 2),
                "goals": round(math.exp(0.5 * (ha + atk + base + dfs + base)), 2),
                "played": fm.played.get(t, 0),
                "source": fm.rating_source(t),
            })
        rows.sort(key=lambda r: r["str"], reverse=True)
        return rows

    def _carry_ratings_up(self, div: str, gm) -> None:
        """Give a club promoted into `div` the rating it earned below.

        A club with no history here would otherwise get the flat promoted-team
        prior. If it played in a division on the same ladder, its real rating
        carries over, shifted by the measured gap between the two rungs.
        """
        lad = leagues.ladder_of(div)
        if lad is None:
            return
        here = set(self.df["Div"])
        # Nearest rung first, so a club that played two divisions down does not
        # override the record it has one division down.
        rungs = sorted((o for o in lad if o != div and o in here),
                       key=lambda o: abs(lad.index(o) - lad.index(div)))
        for other in rungs:
            shift = leagues.rating_shift(other, div)
            if shift is None:
                continue
            try:
                src = model.fit(self.df, other, gm.window, xi=self.xi,
                                as_of=self.as_of, goal_shrink=self.goal_shrink,
                                edge_scale=self.edge_scale)
            except Exception:
                continue
            da, dd = shift
            for t in src.teams:
                if t in gm.attack or t in gm.carried_attack:
                    continue
                if src.played.get(t, 0) < 10:      # too thin to be worth carrying
                    continue
                gm.carried_attack[t] = src.attack[t] + da
                gm.carried_defence[t] = (src.defence[t] - src.mean_defence) \
                    + dd + gm.mean_defence
                gm.carried[t] = other

    # --------------------------------------------------------- name lookups
    # Fuzzy matching has to be tight. Loosen it and "Le Mans" matches "Lens":
    # a newly promoted club silently predicted as a different one, which is the
    # worst failure this tool can have. But similarity alone cannot separate
    # good from bad - "Sheff Utd"/"Sheffield United" (0.72) scores below
    # "Le Mans"/"Lens" (0.73). So common abbreviations are handled by an
    # explicit alias table, and fuzzy stays tight enough to reject all of them.
    FUZZY_CUTOFF = 0.85
    FUZZY_MARGIN = 0.08          # the best match must clearly beat the runner-up

    # Whole-name aliases for clubs whose common name shares nothing useful with
    # the name football-data uses.
    ALIASES = {
        "spurs": "tottenham", "psg": "paris sg", "barca": "barcelona",
        "atletico": "ath madrid", "atletico madrid": "ath madrid",
        "athletic": "ath bilbao", "athletic bilbao": "ath bilbao",
        "inter milan": "inter", "ac milan": "milan", "bayern": "bayern munich",
        "dortmund": "dortmund", "gladbach": "m'gladbach", "juve": "juventus",
        "wolves": "wolves", "man u": "man united", "spurs fc": "tottenham",
    }
    # Token rewrites, applied word by word.
    TOKEN_ALIASES = {"utd": "united", "weds": "wednesday", "nott": "nott'm",
                     "notts": "nott'm", "st.": "st", "sheff": "sheffield"}
    # Noise words dropped only if the name did not match as typed.
    NOISE = {"fc", "afc", "cf", "sc", "ac", "ss", "as", "us", "club",
             "fk", "sk", "bk", "if", "il", "ik", "sv", "sg", "nk", "hnk",
             "ac.", "kv", "kaa", "rsc", "tsv", "scr", "vfb", "vfl"}
    # Nordic and central European spellings the source files keep but a typed
    # name usually drops.
    FOLD = str.maketrans({"ø": "o", "å": "a", "æ": "ae", "ä": "a", "ö": "o",
                          "ü": "u", "é": "e", "í": "i", "á": "a", "š": "s",
                          "č": "c", "ž": "z", "ě": "e", "ř": "r", "ů": "u",
                          "ý": "y", "ň": "n", "ť": "t", "ď": "d", "ł": "l",
                          "ß": "ss"})

    def _variants(self, name: str) -> list[str]:
        """The typed name, then progressively normalised forms of it."""
        key = " ".join(name.strip().casefold().split())
        out = [key, key.translate(self.FOLD)]
        if key in self.ALIASES:
            out.append(self.ALIASES[key])
        key = key.translate(self.FOLD)
        toks = [self.TOKEN_ALIASES.get(t, t) for t in key.split()]
        if toks != key.split():
            out.append(" ".join(toks))
        stripped = [t for t in toks if t not in self.NOISE]
        if stripped and stripped != toks:
            out.append(" ".join(stripped))
        seen, uniq = set(), []
        for v in out:
            if v and v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def teams(self, div: str | None = None) -> list[str]:
        """Every club in one division (or all of them), cached."""
        cache = self._memo("_teams", dict)
        if div not in cache:
            cache[div] = loader.teams(self.df, div)
        return cache[div]

    def _folded_pool(self, div: str | None) -> list[tuple]:
        """(name, div, casefolded, accent-folded) for every club, cached.

        Case-folding a whole division for every name resolved was a third of
        the cost of a slate: two clubs per fixture, forty-odd fixtures, the
        same few hundred strings folded again each time.
        """
        cache = self._memo("_pool", dict)
        if div not in cache:
            cache[div] = [(t, d, t.casefold(), t.casefold().translate(self.FOLD))
                          for d in ([div] if div else self.divs)
                          for t in self.teams(d)]
        return cache[div]

    def resolve(self, name: str, div: str | None = None,
                fuzzy: bool = True) -> tuple[str, str]:
        """Match a typed team name to a real one, and say which league it is in."""
        starts = contains = []
        folded = self._folded_pool(div)
        for key in self._variants(name):
            for col in (2, 3):        # as written, then accent-folded
                exact = [x for x in folded if x[col] == key]
                if exact:
                    return exact[0][0], exact[0][1]
                starts = [x for x in folded if x[col].startswith(key)]
                if len(starts) == 1:
                    return starts[0][0], starts[0][1]
                contains = [x for x in folded if key in x[col]]
                if len(contains) == 1:
                    return contains[0][0], contains[0][1]
            starts = [(x[0], x[1]) for x in starts]
            contains = [(x[0], x[1]) for x in contains]
        names = [x[0] for x in folded]
        if fuzzy:
            close = difflib.get_close_matches(name, names, n=2,
                                              cutoff=self.FUZZY_CUTOFF)
            if close:
                best = difflib.SequenceMatcher(None, name, close[0]).ratio()
                runner = (difflib.SequenceMatcher(None, name, close[1]).ratio()
                          if len(close) > 1 else 0.0)
                if len(close) == 1 or best - runner >= self.FUZZY_MARGIN:
                    hit = next(x for x in folded if x[0] == close[0])
                    return hit[0], hit[1]
        opts = sorted({x[0] for x in (starts or contains)})[:10]
        if not opts:
            opts = difflib.get_close_matches(name, names, n=5, cutoff=0.5)
        hint = ("  Did you mean: " + ", ".join(opts)) if opts else ""
        raise SystemExit("Unknown team: %r%s" % (name, hint))

    def resolve_pair(self, home: str, away: str, div: str | None = None,
                     allow_new: bool = False):
        """Resolve both sides. With allow_new, a name unknown to a named
        division is kept as-is and treated as a newly promoted club."""
        if allow_new and div:
            h, hnew = self._resolve_or_new(home, div)
            a, anew = self._resolve_or_new(away, div)
            return h, a, div, hnew, anew
        h, dh = self.resolve(home, div)
        a, da = self.resolve(away, div)
        if dh != da and div is None:
            # Prefer the league where both names exist, otherwise the home side's.
            both = [d for d in self.divs
                    if h in loader.teams(self.df, d) and a in loader.teams(self.df, d)]
            if both:
                return h, a, both[0], False, False
        return h, a, div or dh, False, False

    def _resolve_or_new(self, name: str, div: str) -> tuple[str, bool]:
        try:
            return self.resolve(name, div, fuzzy=False)[0], False
        except SystemExit:
            return name.strip(), True

    # ------------------------------------------------------------ predicting
    def predict(self, home: str, away: str, div: str | None = None,
                neutral: bool = False, allow_new: bool = False,
                odds=None) -> dict:
        h, a, d, hnew, anew = self.resolve_pair(home, away, div, allow_new)
        ms = self.models(d)
        fm = ms["FT"]
        lam, mu = fm.rates(h, a, neutral)

        # The price, read back into goal rates, acts as a prior. The pure model
        # is kept alongside: once the price is inside the forecast, comparing
        # the forecast to the price is circular.
        mk = None
        if odds is not None and self.market_weight > 0:
            try:
                mk = market.rates_from_row(odds, fm.rho, guess=(lam, mu))
            except Exception:
                mk = None
        blam, bmu = market.blend_rates((lam, mu), mk, self.market_weight)
        scale = (blam / lam, bmu / mu)

        ft = model.score_matrix_from_rates(blam, bmu, fm.rho)
        f1 = self._half(ms.get("1H"), h, a, neutral, scale)
        f2 = self._half(ms.get("2H"), h, a, neutral, scale)
        s = markets.summary(ft, f1, f2)
        s["score_grid"] = markets.score_grid(ft)
        s["score_grid_other"] = markets.score_grid_remainder(ft)

        # what the model said on its own, before the price was folded in
        pure = markets.summary(model.score_matrix_from_rates(lam, mu, fm.rho))
        s["model_result"] = pure["result"]
        s["model_totals"] = pure["totals"]
        s["model_btts"] = pure["btts"]
        s["model_exp_home"], s["model_exp_away"] = lam, mu
        s["market_used"] = mk is not None
        s["market_weight"] = self.market_weight if mk is not None else 0.0
        if mk is not None:
            s["market_exp_home"], s["market_exp_away"] = mk
            mm = markets.summary(model.score_matrix_from_rates(mk[0], mk[1], fm.rho))
            s["market_result"] = mm["result"]

        s["home"], s["away"], s["div"] = h, a, d
        s["pick"] = max(("1", "X", "2"), key=lambda k: s["result"].get(k, 0.0))
        s["home_new"], s["away_new"] = hnew, anew
        s["home_source"] = fm.rating_source(h)
        s["away_source"] = fm.rating_source(a)
        s["home_played"] = fm.played.get(h, 0)
        s["away_played"] = fm.played.get(a, 0)
        s["n_train"] = fm.n_matches
        s["home_adv"] = fm.home_adv
        return s

    @staticmethod
    def _half(hm, h, a, neutral, scale):
        """Half-time matrix, nudged by the same factor the price moved full time."""
        if hm is None:
            return None
        lam, mu = hm.rates(h, a, neutral)
        return model.score_matrix_from_rates(lam * scale[0], mu * scale[1], hm.rho)

    # ------------------------------------------------- cross-league ties
    def can_bridge(self, home: str, away: str) -> bool:
        """True when both clubs are rated and a bridge exists between them."""
        if not self.bridge:
            return False
        try:
            _, hd = self.resolve(home, None, fuzzy=False)
            _, ad = self.resolve(away, None, fuzzy=False)
        except SystemExit:
            return False
        off = self.bridge["offsets"]
        return hd != ad and hd in off and ad in off

    def predict_cross(self, home: str, away: str, neutral: bool = False,
                      comp: str = "") -> dict:
        """Price a tie between clubs from two different domestic leagues.

        Domestic ratings are centred within their own league, so they are only
        comparable once each league's measured strength offset is applied.
        """
        import numpy as np

        h, hdiv = self.resolve(home, None, fuzzy=False)
        a, adiv = self.resolve(away, None, fuzzy=False)
        if hdiv == adiv:
            return self.predict(h, a, hdiv, neutral=neutral)
        if not self.bridge:
            raise SystemExit("No country bridge available; run scratch/fit_bridge.py")
        off = self.bridge["offsets"]
        if hdiv not in off or adiv not in off:
            missing = [d for d in (hdiv, adiv) if d not in off]
            raise SystemExit("No measured bridge for division(s) %s"
                             % ", ".join(missing))

        # raw ratings, because the offsets were calibrated against raw ratings
        mh = model.fit(self.df, hdiv, "FT", xi=self.xi, as_of=self.as_of,
                       goal_shrink=1.0, edge_scale=1.0)
        ma = model.fit(self.df, adiv, "FT", xi=self.xi, as_of=self.as_of,
                       goal_shrink=1.0, edge_scale=1.0)
        h_att, h_def = mh.attack.get(h, 0.0), mh.defence.get(h, mh.mean_defence)
        a_att, a_def = ma.attack.get(a, 0.0), ma.defence.get(a, ma.mean_defence)
        h_def -= mh.mean_defence
        a_def -= ma.mean_defence
        c = self.bridge["intercept"]
        adv = 0.0 if neutral else self.bridge["home_adv"]
        lvl = 0.5 * (mh.base + ma.base)
        sh, sa = off[hdiv], off[adiv]
        loglam = c + lvl + h_att + a_def + sh - sa + adv
        logmu = c + lvl + a_att + h_def + sa - sh
        # Same level/edge split as the domestic model, but with its own tuned
        # constants: European ties want no level shrinkage, because a genuine
        # mismatch between a strong and a weak league really does run up scores.
        gs = self.bridge.get("goal_shrink", 1.0)
        es = self.bridge.get("edge_scale", 1.0)
        base_lvl = self.bridge.get("mean_level", 0.5 * (loglam + logmu))
        level = 0.5 * (loglam + logmu)
        edge = 0.5 * (loglam - logmu)
        level = base_lvl + gs * (level - base_lvl)
        lam = float(np.exp(level + es * edge))
        mu = float(np.exp(level - es * edge))
        rho = 0.5 * (mh.rho + ma.rho)
        ft = model.score_matrix_from_rates(lam, mu, rho)

        s = markets.summary(ft)
        s["home"], s["away"] = h, a
        s["div"] = comp or "%s v %s" % (hdiv, adiv)
        s["home_div"], s["away_div"] = hdiv, adiv
        s["cross"] = True
        s["home_new"] = s["away_new"] = False
        s["home_source"], s["away_source"] = hdiv, adiv
        s["home_played"] = mh.played.get(h, 0)
        s["away_played"] = ma.played.get(a, 0)
        s["market_used"] = False
        s["bridge_offsets"] = (sh, sa)
        return s

    # ----------------------------------------------------------- CAF ties
    def predict_caf(self, home: str, away: str, comp: str = "CAF CL") -> dict:
        """Price a CAF club tie. Names may carry a country code: 'Club (MWI)'.

        A club from a league we load is rated from its own results. A club
        from a federation we do not load is rated as an average club of that
        federation, using the federation's offset measured from CAF matches -
        the only evidence there is about it.
        """
        import re as _re
        import numpy as np
        from . import euro

        if not self.caf_bridge:
            raise SystemExit("No CAF bridge; run scratch/fit_caf.py")
        br = self.caf_bridge
        off, counts = br["offsets"], br.get("matches", {})

        def side(name):
            m = _re.match(r"^(.*?)\s*\(([A-Z]{3})\)\s*$", name.strip())
            club, code = (m.group(1), m.group(2)) if m else (name.strip(), None)
            div = euro.CAF_COUNTRY_DIV.get(code) if code else None
            if div is None and code is None:
                try:
                    _, div = self.resolve(club, None, fuzzy=False)
                except SystemExit:
                    div = None
            if div and div in self.divs:
                try:
                    team = self.resolve(club, div, fuzzy=False)[0]
                    gm = model.fit(self.df, div, "FT", xi=self.xi,
                                   as_of=self.as_of, goal_shrink=1.0,
                                   edge_scale=1.0)
                    if gm.knows(team):
                        return {"name": team, "key": div, "fed": False,
                                "att": gm.attack[team],
                                "def": gm.defence[team] - gm.mean_defence,
                                "base": gm.base, "rho": gm.rho,
                                "code": code, "league": leagues.name(div)}
                except SystemExit:
                    pass
            key = "F-" + (code or "UNK")
            return {"name": club, "key": key, "fed": True, "att": 0.0,
                    "def": 0.0, "base": br.get("fed_base", 0.0), "rho": -0.05,
                    "code": code, "league": "federation " + (code or "unknown"),
                    "ties": counts.get(key, 0)}

        H, A = side(home), side(away)
        sh, sa = off.get(H["key"], 0.0), off.get(A["key"], 0.0)
        lvl = 0.5 * (H["base"] + A["base"])
        c, adv = br["intercept"], br["home_adv"]
        a_ = c + lvl + H["att"] + A["def"] + sh - sa + adv
        b_ = c + lvl + A["att"] + H["def"] + sa - sh
        L, E = 0.5 * (a_ + b_), 0.5 * (a_ - b_)
        es = br.get("edge_scale", 1.0)
        lam, mu = float(np.exp(L + es * E)), float(np.exp(L - es * E))
        ft = model.score_matrix_from_rates(lam, mu, 0.5 * (H["rho"] + A["rho"]))
        s = markets.summary(ft)
        s.update({"home": H["name"], "away": A["name"], "div": comp,
                  "cross": True, "caf": True, "matrix": ft,
                  "home_new": False, "away_new": False,
                  "home_source": H["league"], "away_source": A["league"],
                  "home_fed": H["fed"], "away_fed": A["fed"],
                  "home_fed_ties": H.get("ties"), "away_fed_ties": A.get("ties"),
                  "home_code": H["code"], "away_code": A["code"],
                  "market_used": False, "home_played": 0, "away_played": 0})
        return s

    # ------------------------------------------------------------- schedule
    def _fixtures(self, fixtures_csv: str | None) -> pd.DataFrame:
        """The whole schedule, read from disk at most once per change.

        A slate asks for the schedule once per division, and every one of those
        calls re-read and re-parsed the same feed plus every overlay file -
        thirty round trips to the disk for one answer, and the single largest
        cost in the response. The mtimes are the cache key, so an offline
        `refresh-fixtures` is still picked up without a restart.
        """
        cache = os.path.join(self.root, "fixtures.csv")
        over = fixtures.default_overlay_dir()
        paths = [p for p in (fixtures_csv, cache) if p] + \
            sorted(glob.glob(os.path.join(over, "*.csv")))
        key = tuple((p, os.path.getmtime(p)) if os.path.exists(p) else (p, None)
                    for p in paths)
        got = self.__dict__.get("_fx_cache")
        if got is None or got[0] != key:
            got = self.__dict__["_fx_cache"] = (
                key, fixtures.load_any(fixtures_csv, cache))
        return got[1]

    def schedule(self, div: str | None = None, days: int = 14,
                 fixtures_csv: str | None = None) -> pd.DataFrame:
        fx = self._fixtures(fixtures_csv)
        return fixtures.upcoming(self.df, fx, div, days, self.as_of,
                                 remaining_fn=self._remaining)

    def _remaining(self, _df, div: str) -> pd.DataFrame:
        """The round-robin remainder for one division, computed once.

        It is a function of results already on disk, and `self.df` does not
        change under us. Handed out as a copy so a caller that annotates or
        filters it cannot corrupt the next reader's view.
        """
        cache = self._memo("_rem", dict)
        if div not in cache:
            cache[div] = fixtures.remaining(self.df, div)
        return cache[div].copy()

    def form(self, team: str, div: str | None = None, n: int = 6) -> pd.DataFrame:
        t, d = self.resolve(team, div)
        m = self.df[(self.df["Div"] == d) &
                    ((self.df["HomeTeam"] == t) | (self.df["AwayTeam"] == t))]
        return m.sort_values("Date").tail(n)


# ------------------------------------------------------- the country bridge
def load_bridge(path=None):
    """Country strength offsets measured from real cross-league results."""
    import json
    path = path or os.path.join(os.path.dirname(__file__), "bridge.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_caf_bridge(path=None):
    """Country and federation offsets measured from CAF club competitions."""
    import json
    path = path or os.path.join(os.path.dirname(__file__), "bridge_caf.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tie_outcome(ft, leg1_this_home: int, leg1_this_away: int) -> dict:
    """Chance of winning a two-legged tie, from the second leg's score matrix.

    `ft` is the second-leg matrix with rows as the second-leg home side.
    leg1_this_home / leg1_this_away are the goals the second-leg home and away
    sides scored in the first leg. Level on aggregate is reported separately
    rather than resolved, because it goes to extra time and penalties and the
    engine does not model either.
    """
    import numpy as np
    n = ft.shape[0]
    h = np.arange(n)[:, None] + leg1_this_home
    a = np.arange(n)[None, :] + leg1_this_away
    return {"win": float(ft[h > a].sum()), "level": float(ft[h == a].sum()),
            "lose": float(ft[h < a].sum())}
