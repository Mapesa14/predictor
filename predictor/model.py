"""Dixon-Coles bivariate Poisson goal model with exponential time decay.

One model is fitted per league and per scoring window (full time, first half,
second half).  A fitted model turns any fixture into a matrix of scoreline
probabilities, which every betting market is then read off.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 12
_RIDGE = 0.02          # shrinks thin-sample teams toward the league average

# A side newly promoted into a top division, relative to that league's average.
# Measured over 45 promoted team-seasons in 11 top divisions by
# scratch/promoted.py: they score 0.79x and concede 1.16x the league average.
PROMOTED_ATTACK = -0.232
PROMOTED_DEFENCE = 0.150


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score dependence correction."""
    t = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return np.clip(t, 1e-6, None)


@dataclass
class GoalModel:
    div: str
    window: str                      # "FT" | "1H" | "2H"
    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_adv: float
    rho: float
    xi: float
    n_matches: int
    played: dict[str, int] = field(default_factory=dict)
    base: float = 0.0                # league mean log goal rate
    mean_log_rate: float = 0.0       # average log rate over all pairings
    mean_defence: float = 0.0        # league average defence rating
    goal_shrink: float = 1.0         # <1 pulls total goals toward the league mean
    edge_scale: float = 1.0          # <1 pulls the home/away split toward even
    target: str = "goals"            # "goals" | "sot" | "shots"
    conversion: float = 1.0          # goals per unit of target, 1 for goals
    # Ratings for clubs with no history in this division, carried across from
    # another rung of the same promotion ladder. Kept apart from `attack` and
    # `defence` so `teams`, `knows` and the ratings table stay honest about
    # who has actually played here.
    carried: dict = field(default_factory=dict)          # team -> source div
    carried_attack: dict = field(default_factory=dict)
    carried_defence: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- rates
    def rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        # A team with no history in this division is treated as newly promoted
        # rather than as an average side, which is what the data says it is.
        ah, dh = self._team(home)
        aa, da = self._team(away)
        adv = 0.0 if neutral else self.home_adv
        log_lam = self.base + ah + da + adv
        log_mu = self.base + aa + dh
        # Split into "how many goals" and "who scores them", then shrink each
        # toward the league average.  Raw ratings are over-dispersed on totals.
        level = 0.5 * (log_lam + log_mu)
        edge = 0.5 * (log_lam - log_mu)
        level = self.mean_log_rate + self.goal_shrink * (level - self.mean_log_rate)
        edge = self.edge_scale * edge
        return float(np.exp(level + edge)), float(np.exp(level - edge))

    def _team(self, t: str) -> tuple[float, float]:
        """Attack and defence for a club: its own record here, else the rating
        carried from another rung of the ladder, else the promoted prior."""
        if t in self.attack:
            return self.attack[t], self.defence[t]
        if t in self.carried_attack:
            return self.carried_attack[t], self.carried_defence[t]
        return PROMOTED_ATTACK, self.mean_defence + PROMOTED_DEFENCE

    def rating_source(self, t: str) -> str:
        """Where this club's rating came from: this division, another, or the prior."""
        if t in self.attack:
            return self.div
        return self.carried.get(t, "prior")

    # --------------------------------------------------------------- matrix
    def score_matrix(self, home: str, away: str, neutral: bool = False,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.rates(home, away, neutral)
        return score_matrix_from_rates(lam, mu, self.rho, max_goals)

    def knows(self, team: str) -> bool:
        return team in self.attack


def score_matrix_from_rates(lam: float, mu: float, rho: float,
                            max_goals: int = MAX_GOALS) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals + 1), lam)
    a = poisson.pmf(np.arange(max_goals + 1), mu)
    m = np.outer(h, a)
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum()


# ------------------------------------------------------------------- fitting
# What a model can be fitted on. Goals are what we ultimately predict; shots
# measure the same underlying team strength with far less noise per match.
TARGETS = {
    "goals": ("FTHG", "FTAG"),
    "sot": ("HST", "AST"),
    "shots": ("HS", "AS"),
}


def _prepare(df, window, target="goals"):
    if target != "goals":
        if window != "FT":
            raise ValueError("shot counts exist only for the full match")
        ch, ca = TARGETS[target]
        return (np.clip(df[ch].to_numpy(float), 0, None),
                np.clip(df[ca].to_numpy(float), 0, None))
    if window == "FT":
        hg, ag = df["FTHG"].to_numpy(int), df["FTAG"].to_numpy(int)
    elif window == "1H":
        hg, ag = df["HTHG"].to_numpy(int), df["HTAG"].to_numpy(int)
    elif window == "2H":
        hg = (df["FTHG"] - df["HTHG"]).to_numpy(int)
        ag = (df["FTAG"] - df["HTAG"]).to_numpy(int)
    else:
        raise ValueError(window)
    return np.clip(hg, 0, None), np.clip(ag, 0, None)


def fit(df, div: str, window: str = "FT", xi: float = 0.0018, as_of=None,
        goal_shrink: float = 1.0, edge_scale: float = 1.0,
        target: str = "goals") -> GoalModel:
    """Fit one league's model on matches played before `as_of`."""
    if target not in TARGETS:
        raise ValueError("unknown target %r" % target)
    d = df[df["Div"] == div]
    if window in ("1H", "2H"):
        d = d.dropna(subset=["HTHG", "HTAG"])
    if target != "goals":
        cols = list(TARGETS[target])
        if not set(cols).issubset(d.columns):
            raise ValueError("%s: no %s data available" % (div, target))
        d = d.dropna(subset=cols)
    if as_of is not None:
        when = d["known_at"] if "known_at" in d.columns else d["Date"]
        d = d[when < as_of]
    d = d.reset_index(drop=True)
    if len(d) < 40:
        raise ValueError(f"{div}: only {len(d)} matches, not enough to fit")

    ref = as_of if as_of is not None else d["Date"].max()
    age = (ref - d["Date"]).dt.days.to_numpy(float)
    w = np.exp(-xi * np.clip(age, 0, None))

    teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = d["HomeTeam"].map(idx).to_numpy(int)
    ai = d["AwayTeam"].map(idx).to_numpy(int)
    hg, ag = _prepare(d, window, target)
    base = np.log(max((w * (hg + ag)).sum() / (2 * w.sum()), 0.05))
    # Goals per shot, on the same time-decayed sample: converts a shot-model
    # rate into an expected-goals rate.
    if target == "goals":
        conversion = 1.0
    else:
        gh, ga = _prepare(d, "FT", "goals")
        conversion = float((w * (gh + ga)).sum() / max((w * (hg + ag)).sum(), 1e-9))

    # params: attack[0..n-2] (last is fixed by the zero-sum constraint),
    #         defence[0..n-1], home advantage, rho
    def unpack(p):
        att = np.empty(n)
        att[:-1] = p[:n - 1]
        att[-1] = -att[:-1].sum()
        dfn = p[n - 1:2 * n - 1]
        return att, dfn, p[-2], p[-1]

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    mean_w = w.sum() / len(w)

    def _parts(p):
        att, dfn, adv, rho = unpack(p)
        lam = np.clip(np.exp(base + att[hi] + dfn[ai] + adv), 1e-6, 25.0)
        mu = np.clip(np.exp(base + att[ai] + dfn[hi]), 1e-6, 25.0)
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-6, None)
        return att, dfn, adv, rho, lam, mu, tau

    def nll(p):
        att, dfn, adv, rho, lam, mu, tau = _parts(p)
        ll = (hg * np.log(lam) - lam) + (ag * np.log(mu) - mu) + np.log(tau)
        pen = _RIDGE * (np.sum(att ** 2) + np.sum(dfn ** 2))
        return -float(np.sum(w * ll)) + pen * mean_w

    def grad(p):
        att, dfn, adv, rho, lam, mu, tau = _parts(p)
        # d(tau)/d(lam), d(tau)/d(mu), d(tau)/d(rho) - non-zero only on low scores
        dt_l = np.zeros_like(lam); dt_m = np.zeros_like(lam); dt_r = np.zeros_like(lam)
        dt_l[m00] = -mu[m00] * rho; dt_m[m00] = -lam[m00] * rho
        dt_r[m00] = -lam[m00] * mu[m00]
        dt_l[m01] = rho;            dt_r[m01] = lam[m01]
        dt_m[m10] = rho;            dt_r[m10] = mu[m10]
        dt_r[m11] = -1.0
        # chain rule through log(lam) = base + att + dfn (+adv), so d/dparam = A
        A = w * (hg - lam + lam * dt_l / tau)
        B = w * (ag - mu + mu * dt_m / tau)
        g_att = np.bincount(hi, A, n) + np.bincount(ai, B, n)
        g_dfn = np.bincount(ai, A, n) + np.bincount(hi, B, n)
        g_adv = float(A.sum())
        g_rho = float(np.sum(w * dt_r / tau))
        # penalty, and the zero-sum constraint on the last attack parameter
        g_att = -g_att + mean_w * 2 * _RIDGE * att
        g_dfn = -g_dfn + mean_w * 2 * _RIDGE * dfn
        free = g_att[:-1] - g_att[-1]
        return np.concatenate([free, g_dfn, [-g_adv], [-g_rho]])

    p0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [0.0]])
    # rho corrects the 0-0/1-0/0-1/1-1 cluster, which is a goals phenomenon;
    # shot counts never sit down there, so it is pinned off for those targets.
    rho_bound = (-0.25, 0.25) if target == "goals" else (0.0, 0.0)
    bounds = ([(-2.0, 2.0)] * (n - 1) + [(-2.0, 2.0)] * n +
              [(-1.0, 1.0), rho_bound])
    res = minimize(nll, p0, jac=grad, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 800, "maxfun": 200000, "ftol": 1e-10})
    att, dfn, adv, rho = unpack(res.x)

    all_lvl = [0.5 * (2 * base + att[i] + dfn[j] + att[j] + dfn[i] + adv)
               for i in range(n) for j in range(n) if i != j]
    counts = d["HomeTeam"].value_counts().add(
        d["AwayTeam"].value_counts(), fill_value=0).astype(int).to_dict()
    return GoalModel(
        div=div, window=window, teams=teams,
        attack={t: float(att[i]) for t, i in idx.items()},
        defence={t: float(dfn[i]) for t, i in idx.items()},
        home_adv=float(adv), rho=float(rho), xi=xi, target=target,
        conversion=conversion,
        n_matches=len(d), played={t: int(counts.get(t, 0)) for t in teams},
        base=float(base), mean_log_rate=float(np.mean(all_lvl)),
        mean_defence=float(np.mean(dfn)),
        goal_shrink=goal_shrink, edge_scale=edge_scale,
    )


def fit_all(df, div: str, xi: float = 0.0018, as_of=None,
            goal_shrink: float = 1.0, edge_scale: float = 1.0,
            weights: dict | None = None) -> dict:
    """Fit the full-time, first-half and second-half models for one league."""
    out = {}
    for window in ("FT", "1H", "2H"):
        try:
            out[window] = fit_blended(df, div, window, xi=xi, as_of=as_of,
                                      goal_shrink=goal_shrink,
                                      edge_scale=edge_scale, weights=weights)
        except Exception:
            pass
    return out


def strength_table(m: GoalModel) -> list[tuple[str, float, float, float, int]]:
    """Per-team attack / defence multipliers and a single overall rating."""
    rows = []
    for t in m.teams:
        atk = float(np.exp(m.attack[t]))
        dfc = float(np.exp(m.defence[t]))
        rows.append((t, atk, dfc, atk / dfc, m.played.get(t, 0)))
    return sorted(rows, key=lambda r: -r[3])


# ------------------------------------------------------- goals + shots blend
@dataclass
class BlendedModel:
    """Goal rates read from goals and from shots, combined in log space.

    Goals are the target but a noisy per-match measure of team strength; shots
    on target say much the same thing with far less variance. Both models are
    fitted on the same history and their rates blended, then converted back
    into a scoreline matrix using the goal model's Dixon-Coles rho.
    """
    goals: GoalModel
    shots: dict                      # target name -> GoalModel
    weights: dict                    # target name -> weight in the blend

    # ---- the parts of GoalModel that callers reach for -------------------
    @property
    def div(self):
        return self.goals.div

    @property
    def window(self):
        return self.goals.window

    @property
    def teams(self):
        return self.goals.teams

    @property
    def attack(self):
        return self.goals.attack

    @property
    def defence(self):
        return self.goals.defence

    @property
    def home_adv(self):
        return self.goals.home_adv

    @property
    def rho(self):
        return self.goals.rho

    @property
    def n_matches(self):
        return self.goals.n_matches

    @property
    def played(self):
        return self.goals.played

    @property
    def mean_defence(self):
        return self.goals.mean_defence

    @property
    def carried(self):
        return self.goals.carried

    @property
    def carried_attack(self):
        return self.goals.carried_attack

    @property
    def carried_defence(self):
        return self.goals.carried_defence

    def rating_source(self, t: str) -> str:
        return self.goals.rating_source(t)

    def _team(self, t: str):
        return self.goals._team(t)

    def knows(self, team: str) -> bool:
        return self.goals.knows(team)

    # ---- rates ------------------------------------------------------------
    def rates(self, home: str, away: str, neutral: bool = False):
        lam, mu = self.goals.rates(home, away, neutral)
        wl = [(self.weights["goals"], np.log(lam), np.log(mu))]
        for name, m in self.shots.items():
            w = self.weights.get(name, 0.0)
            if w <= 0 or not (m.knows(home) or m.knows(away)):
                continue
            sl, sm = m.rates(home, away, neutral)
            wl.append((w, np.log(sl * m.conversion), np.log(sm * m.conversion)))
        tot = sum(w for w, _, _ in wl)
        if tot <= 0:
            return lam, mu
        lh = sum(w * a for w, a, _ in wl) / tot
        la = sum(w * b for w, _, b in wl) / tot
        return float(np.exp(lh)), float(np.exp(la))

    def score_matrix(self, home: str, away: str, neutral: bool = False,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.rates(home, away, neutral)
        return score_matrix_from_rates(lam, mu, self.goals.rho, max_goals)


def fit_blended(df, div: str, window: str = "FT", xi: float = 0.0018,
                as_of=None, goal_shrink: float = 1.0, edge_scale: float = 1.0,
                weights: dict | None = None):
    """Fit the goal model, plus a shot model for every target given weight."""
    weights = weights or {"goals": 1.0}
    g = fit(df, div, window, xi=xi, as_of=as_of, goal_shrink=goal_shrink,
            edge_scale=edge_scale, target="goals")
    if window != "FT" or not any(weights.get(t, 0) > 0 for t in ("sot", "shots")):
        return g
    shots = {}
    for name in ("sot", "shots"):
        if weights.get(name, 0.0) <= 0:
            continue
        try:
            shots[name] = fit(df, div, window, xi=xi, as_of=as_of,
                              goal_shrink=goal_shrink, edge_scale=edge_scale,
                              target=name)
        except (ValueError, KeyError):
            pass
    if not shots:
        return g
    return BlendedModel(goals=g, shots=shots, weights=dict(weights))
