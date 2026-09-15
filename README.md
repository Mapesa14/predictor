# Football score predictor

Predicts scorelines and the betting markets that fall out of them, for the top
European leagues, from historical results in
`D:\Downloads July 2026\SoccerData` (football-data.co.uk format).

```bash
python predict.py predict "Arsenal vs Chelsea" -d 2026-08-30 -b
```

```
ARSENAL vs CHELSEA
Premier League
30 Aug 2026

                      PREDICTION        PROBABILITY
----------------------------------------------------
Home Win              Arsenal              61%
Draw                                       23%
Away Win              Chelsea              16%

Double Chance
1X                                         84%
...
EXPECTED GOALS
Arsenal: 1.96
Chelsea: 0.92

MOST LIKELY SCORE
1 - 1   (11%)

BEST BETS
  Arsenal to score           86%
  Arsenal or Draw            84%
  Over 1.5 goals             79%
  Under 3.5 goals            67%
  Arsenal to win             61%
```

## Data

14,788 matches, 2023/24 to 2025/26, de-duplicated across the download folders.
The ten leagues below are treated as the top tier of Europe; five more English
divisions are loaded too and can be used with `-l`.

| Code | League | Country |
|------|--------|---------|
| E0 | Premier League | England |
| SP1 | La Liga | Spain |
| I1 | Serie A | Italy |
| D1 | Bundesliga | Germany |
| F1 | Ligue 1 | France |
| P1 | Primeira Liga | Portugal |
| N1 | Eredivisie | Netherlands |
| B1 | Jupiler Pro League | Belgium |
| T1 | Super Lig | Turkey |
| G1 | Super League Greece | Greece |

Also loaded: `SC0` Scottish Premiership, `E1`–`E3` and `EC` (English tiers 2–5).
Run `python predict.py leagues` to see exactly what is on disk.

## How it predicts

A **Dixon-Coles bivariate Poisson** model, fitted separately for each league.

1. Every team gets an attack and a defence rating. The home side's goal rate is
   `exp(base + attack_home + defence_away + home_advantage)`, the away side's is
   `exp(base + attack_away + defence_home)`.
2. Recent matches count for more: each match is weighted `exp(-xi * days_ago)`.
3. A Dixon-Coles `rho` term corrects the dependence between the two scores at
   0-0, 1-0, 0-1 and 1-1, which independent Poissons get wrong.
4. The same fit is run a second time on **shots on target**, and the two goal
   rates are blended in log space. Goals are what we predict but a noisy
   per-match measure of strength; shots say much the same thing with far less
   variance.
5. Raw ratings are over-dispersed on totals, so the goal *level* is shrunk
   toward the league average (`goal_shrink`) and the home/away *split* is
   scaled (`edge_scale`). Both were tuned out-of-sample.
6. A club with no history in this division inherits its rating from the
   division it came from, shifted by the measured gap between the two.
7. The result is a 13x13 matrix of scoreline probabilities. **Every market is
   read off that one matrix**, so the card can never contradict itself.

Three models are fitted per league: full time, first half, and second half.
Half-time markets come from the first-half model; half-time/full-time from the
two halves convolved. Shot data exists only for the full match, so the half
models run on goals alone.

### Shots on target

Shots are 99.9% populated in these files and were going unused. A team's past
shots on target predict its next match's goals slightly better than its past
goals do (r 0.272 vs 0.258), and blending the two views is better than either
alone — shots *by themselves* score 0.9840, worse than goals at 0.9826, while
an equal blend scores **0.9778**. Weight it with `--sot-weight` (0 turns it
off). Total shots, tried alongside, added nothing beyond shots on target.

### Carrying a promoted club's rating up

A club new to its division used to get a flat league-wide prior. Where it
played a division we also load, its real rating carries across instead, shifted
by the gap between the two rungs. Those gaps are measured from clubs that
actually moved (`scratch/ladder.py`, 48 movers):

| Step | Attack | Defence |
|------|--------|---------|
| Premier League / Championship | 0.608 (se 0.085) | −0.633 (se 0.135) |
| Championship / League One | 0.336 (se 0.046) | −0.420 (se 0.046) |
| League One / League Two | 0.206 (se 0.052) | −0.228 (se 0.065) |
| League Two / National League | 0.275 (se 0.075) | −0.171 (se 0.088) |

On the 82 matches where a newcomer could be traced, carrying the rating beat
the flat prior on the result, and the improvement was monotone in how much
weight it got: 1X2 log-loss 1.1135 at 0% carry, 1.0637 at 50%, **1.0384** at
100%; RPS 0.2469 → 0.2217. Totals moved the other way (0.6929 → 0.7062), the
same trade the flat prior shows. Turn it off with `--no-ladder`.

This only fires where both rungs are loaded, which today means England alone.
Download `SP2`, `D2`, `F2`, `I2` and `N2` from football-data and the same
machinery covers the other leagues — the offsets would need re-measuring per
ladder.

### The closing price as a prior

The bookmakers' price is a better forecast than this model, so rather than
compete with it the price is read back into the model's own currency. Its
de-vigged 1X2 and over/under probabilities are inverted into the pair of goal
rates whose scoreline matrix reproduces them, and those rates are blended with
the model's. The card stays one distribution, so no two markets on it can
contradict each other. The round trip costs about 0.3 percentage points of
accuracy at the median.

De-vigging uses **Shin's method**, which corrects the favourite-longshot bias by
lifting the favourite and shading long prices. Measured over 7,783 matches it
beats flat normalisation: log-loss 0.9524 against 0.9536.

The uncomfortable result, and it should be said plainly: **on match results the
model adds nothing on top of the price.** Sweeping the blend weight from 0 to 1,
1X2 log-loss falls monotonically all the way to pure market.

| Market weight | 1X2 | RPS | Accuracy | O2.5 | BTTS |
|---|---|---|---|---|---|
| 0.0 (model only) | 0.9739 | 0.1963 | 52.7% | 0.6776 | 0.6895 |
| 0.7 | 0.9586 | 0.1917 | 54.2% | 0.6729 | **0.6870** |
| 0.9 (default) | 0.9568 | 0.1912 | 54.3% | 0.6729 | 0.6873 |
| 1.0 (price only) | **0.9563** | **0.1910** | **54.4%** | 0.6731 | 0.6876 |

Totals peak at 0.8 and BTTS at 0.7, so the model does contribute a little on
goals — which makes sense, since the price quotes only one goal line and the
model has the whole distribution. The default is 0.9. Set `--market-weight 0`
for the model's unaided view.

Two things keep this honest. The pure-model probabilities are always computed
and kept (`model_result`, `model_exp_home`), and the value-bet calculation uses
**only** those: a forecast that already contains the price cannot be used to
find value against it. A test pins that down, because the regression would
silently zero out every edge and look like good news.

Where a fixture has no price, the blend falls through to the model alone — which
is the whole reason the model still matters, along with the correct scores,
handicaps and half-time markets the price does not quote at this granularity.

### What was measured and rejected

**Rest and fixture congestion.** Raw, a team on three days' rest or fewer looks
*better* — it scores 1.17x as many goals (p=0.001). That is confounding, not
signal: congested fixture lists belong to the clubs in Europe, who are strong.
Held against the model's own expected goals as an offset, every rest coefficient
collapses to nothing (p ≥ 0.29, `scratch/rest2.py`). It is not in the model.
Worth revisiting only with European fixtures in the feed, since domestic dates
alone make a Thursday night in the Europa League look like a free week.

### Parameters, and how they were chosen

Every knob is grid-searched against walk-forward out-of-sample log-loss over
~5,900 matches: `scratch/tune.py` for the decay, `tune3.py` for the shrinks,
`tune_shots.py` for the shot weight, `retune.py` for the shrinks again once
shots were in.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `xi` | 0.0018 | time decay per day, a match a year old counts ~52% |
| `goal_shrink` | 0.40 | pulls total goals toward the league mean |
| `edge_scale` | 1.10 | sharpens the home/away split |
| `sot` weight | 1.0 | shots on target, weighted equally with goals |
| `_RIDGE` | 0.02 | shrinks thin-sample teams toward average |

Shrinking the goal level was the first real fix. Raw ratings were badly
over-dispersed on totals; at `goal_shrink = 1.0` the model said 84% for over 2.5
on fixtures that came in at 74%. Tuned, out-of-sample log-loss on over/under 2.5
fell from 0.6938 to 0.6808 and on BTTS from 0.7028 to 0.6914.

These two knobs had to be re-tuned once shots were blended in, and `edge_scale`
moved in a direction worth noting: from 0.90 to **1.10**. Under goals alone the
model had to be flattened to stay honest; with shots steadying the rates it is
now better to sharpen the home/away split than to soften it. The calibration
table caught this first — strong favourites were coming in at 86% when the
model said 74% — and the grid search confirmed it. Override with
`--goal-shrink` / `--edge-scale` (1.0 switches each off).

## Markets covered

Read off the full-time matrix: 1X2, double chance, draw no bet, correct score,
over/under 0.5 to 5.5, team totals, both teams to score, clean sheet, win to
nil, odd/even goals, winning margin, Asian handicap (-2.0 to +2.0 including
quarter lines), European handicap, and result-plus-goals combinations.

From the half models: half-time result, half-time totals and BTTS, half-time /
full-time, and which half has more goals.

`predict --full` prints all of them; the default card prints the ones people
actually bet.

## Commands

```bash
python predict.py leagues                          # what data is loaded
python predict.py predict "Real Madrid" "Barcelona"
python predict.py predict "Arsenal vs Chelsea" --full --best
python predict.py predict "Inter" "Milan" --json
python predict.py table -l I1                      # attack/defence ratings
python predict.py form "Bayern Munich"             # recent results
python predict.py fixtures --top10                 # what is still to play
python predict.py slate -l E0 --limit 10           # predict a whole round
python predict.py backtest --out backtest.csv      # accuracy and calibration
python predict.py brief --start 2026-08-30 --end 2026-08-31 -o slate.pdf
```

Useful flags: `-l/--league` to disambiguate a team name, `--neutral` to drop
home advantage, `--as-of YYYY-MM-DD` to predict as the model would have on a
past date (this is what makes the backtest honest), `--xi` to override decay,
`--sot-weight 0` to fall back to goals only, `--no-ladder` to stop promoted
clubs inheriting their lower-division rating.

## Fixtures

The results files contain no future fixtures, so the schedule comes from, in
order:

1. `--fixtures path.csv` — any CSV with `Div,Date,HomeTeam,AwayTeam`.
2. `python predict.py refresh-fixtures` — downloads football-data.co.uk's
   fixtures feed to `<data>/fixtures.csv`. Needs network, and is never run
   automatically. The feed covers the coming few days across ~22 divisions.
3. **The round-robin remainder**, worked out offline: in a league where everyone
   plays everyone home and away, any pairing not yet played is still to come.
   These have no date (`TBD`) and are ordered into rounds so a team appears once
   per round. Split-format leagues (Scotland, Belgium, Greece) are flagged,
   because their remaining pairings are not a reliable schedule.

Whichever of 1–2 is used, `data/manual/fixtures/*.csv` is merged on top. That
overlay exists because `refresh-fixtures` rewrites the cache wholesale, and the
feed is European: without it, a competition outside Europe drops straight to
level 3 and loses its dates, kick-off times and prices.

### API-Football fixtures and cups

The feed publishes once or twice a week and carries no cups, so a midweek round
or a League Cup night can be missing from it. With `LIVE_API_KEY` set:

```bash
python predict.py refresh-fixtures-api --days 2
```

asks API-Football for today and tomorrow — **one request per day**, charged to
the same 100-a-day budget as live scores — and writes
`data/manual/fixtures_api.csv`. The service does this itself on every refresh
cycle. The file only fills gaps: when a fixture is also in the feed or a league's
own site, that copy wins (the feed has the closing price, the site the kick-off).

Club names are matched strictly: a hand-checked alias
(`service/fixtures_api.ALIASES`), then the exact name, then the name with one
generic word (Town, FC, Utd…) dropped — and only if exactly one club is left.
Anything else is reported and left out, never guessed. So are divisions we hold
no results for and leagues whose results are more than 150 days old.

A cup tie is shown under its competition, never its league. Both clubs in one
division: priced on that division. English clubs in different divisions: priced
on the higher one, with the lower division's price alongside, because the two
scales can disagree by ten points or more for the underdog and that spread is
the honest uncertainty. Any other cross-division tie is not priced. Cup ties
stay **out of the tips lists and the public record** — rotated line-ups make a
favourite least reliable where a short list leans on it, and no cup results are
loaded to settle against.

### Tanzania

The NBC Premier League has no feed behind it. openfootball's file stops in June
2026, football-data has never covered Africa, and the aggregators kept serving
the *previous* Tanzanian season well after this one kicked off. The league's own
site is the authority:

```bash
python predict.py refresh-tanzania
```

That reads `ligikuu.co.tz`, writes results to `data/manual/TZ1.csv` and dated
fixtures to `data/manual/fixtures/TZ1.csv`, then rebuilds the league CSV. Two
rules it will not bend: results are **merged**, never replaced (the homepage
publishes a window, not the season — it carried eight of the ten results on
file), and an unrecognised club name **stops the import** rather than being
matched to the nearest thing. Kick-off times are published in East Africa Time
and the service knows it; every other source is read as UK time, which would
put a 16:00 Dar es Salaam kick-off on the card at 18:00. The site publishes no
half-time scores, so TZ1 cards carry no half-time markets and say why.

## The public record

Backtests are self-reported. The record is not: it writes down what was
predicted **before** kick-off and never touches it again.

```bash
python predict.py record-publish      # run this on a schedule, before the games
python predict.py record              # predicted, then what happened
python predict.py record-verify       # has the file been edited?
```

Three rules make it worth something:

- **A fixture that has already started is refused.** So is one with no
  published kick-off time — without a time there is nothing for the prediction
  to predate. Both refusals are counted and reported, never silent.
- **No row is ever rewritten.** Publishing again adds new fixtures and leaves
  existing ones exactly as they were, so the command is safe on a cron.
- **Outcomes are never stored.** Settlement is a join against the same results
  the engine fits on, recomputed on every read, so there is no outcome column
  anyone could quietly correct.

Every row carries a SHA-256 of itself chained to the hash of the row before it.
Edit a probability, delete a loss, reorder the file, and `record-verify` names
the first row that breaks — "unedited" is checkable rather than promised.

The store is an append-only CSV at `data/record/predictions.csv`. In a
deployment set `RECORD_ROOT` to a persistent volume: it is the one piece of
state that cannot be regenerated if it is lost.

Accuracy comes from the same `backtest.score` and `backtest.calibration` the
walk-forward evaluation uses, and the model is compared to the closing price
only on the rows that carry one — scoring the model on everything and the price
on its own subset is the oldest way to flatter a model.

Once fixtures are loaded, `slate` predicts the lot:

```bash
python predict.py slate --top10 --days 5
```

### Promoted teams

The fixtures feed runs ahead of the results, so a new season brings clubs with
no history in that division. Two rules keep this honest.

**A club is never silently swapped for another.** Batch prediction does no fuzzy
matching at all; interactive prediction uses a tight cutoff plus an explicit
alias table. Similarity alone cannot do this job — `Sheff Utd`/`Sheffield
United` scores 0.72, *below* `Le Mans`/`Lens` at 0.73 — so abbreviations are
resolved deterministically and anything inexact is reported back
(`Read 'Man Utd' as 'Man United'.`). Before this, `Le Mans` was predicted as
Lens, `Amedspor` as Pendikspor, and `Erzurumspor` as Bodrumspor.

**A club with no history is rated as newly promoted, not as average.** Measured
over 45 promoted team-seasons across 11 top divisions (`scratch/promoted.py`),
a promoted side scores **0.79x** and concedes **1.16x** the league average
(attack -0.232, se 0.031; defence +0.150, se 0.027 — every division negative on
attack, 9 of 11 positive on defence). Those fixtures are predicted and marked
`*` rather than dropped, and the card names the side that got the prior.

Worth knowing: on the 35 matches in this dataset where the model met a team new
to its division, the prior beat the neutral assumption on the result (1X2
log-loss 1.028 vs 1.057, RPS 0.192 vs 0.202) but was slightly worse on totals
(0.765 vs 0.746). 35 matches settles nothing on its own — the 45-season
measurement is the real evidence and the validation is merely consistent with
it. `scratch/validate_promoted.py` reruns both.

## How good is it?

Walk-forward: fit only on matches played before each fixture, refit weekly,
5,948 predictions across the ten leagues.

| | log-loss (1X2) | RPS | accuracy |
|---|---|---|---|
| Goals only (where this started) | 0.9826 | 0.1988 | 52.4% |
| Shots and the ladder, no price | 0.9739 | 0.1963 | 52.7% |
| **Default, with the price at 0.9** | **0.9568** | **0.1912** | **54.3%** |
| Bookmaker closing consensus | 0.9558 | 0.1909 | — |

Two separate results, and they should not be run together. Without any price,
shots and the promotion ladder closed **a third** of the gap to the closing
line, 0.0257 down to 0.0170, on data already sitting in these files. Folding in
the price then closes almost all of the rest — but that is the price doing the
work, not the model. Goals markets: log-loss 0.6729 on over/under 2.5 and
0.6873 on BTTS.

Read that honestly: **the model is a few percent worse than the closing line.**
That is the expected result. The closing line prices in team news, lineups,
injuries, suspensions, transfers, motivation, and European fixture congestion —
none of which is in a results CSV.

So the backtest also reports what happens if you bet where the model disagrees
with the price. Betting every selection with a 5%+ modelled edge lost **-12.2%
on flat stakes** over 3,761 bets. That is better than the -15.8% the goals-only
model managed, and it is still a losing proposition. That is the number to trust: this tool is
good for understanding a fixture and sizing expectations, and it is not an
edge over the market. Anything that claims otherwise on this data is fitting
noise.

Where it is genuinely well behaved is calibration. Across probability buckets,
predicted frequency tracks realised frequency closely on both results and
goals — for home wins, the 0.3-0.4 bucket came in at 0.362 and the 0.7-0.8
bucket at 0.756 over 1,564 matches. So the percentages on the card mean what
they say, even though they are not sharp enough to beat a price.
`python predict.py backtest` prints both calibration tables in full.

## Layout

```
predict.py              entry point
predictor/
  leagues.py            division codes and names
  loader.py             read and de-duplicate the CSVs
  model.py              Dixon-Coles fit, analytic gradient, goals/shots blend
  market.py             de-vigging, price -> goal rates, blending
  markets.py            score matrix -> every betting market
  fixtures.py           schedule sources
  engine.py             caching, team-name resolution
  report.py             the match card
  brief.py              a slate of fixtures as a PDF
  backtest.py           walk-forward evaluation, calibration, value bets
  cli.py                argument parsing
tests/                  176 tests: market consistency, name resolution, and
                        recovery of known ratings from simulated seasons
scratch/                tuning scripts and their output
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pandas scipy reportlab pytest
.venv/Scripts/python.exe predict.py leagues
.venv/Scripts/python.exe -m pytest tests -q
```

Point at a different data folder with `--data` or the `SOCCER_DATA` environment
variable.
