# Build prompt — football prediction engine, web app and mobile app

Paste everything below into the agent you want to build this. It is written as
an instruction to that agent. Every number in it was measured, not assumed; the
"Measured constants" section is the source of truth and must not be invented
around.

---

## 0. What you are building

A football prediction engine plus a web app and a mobile app. It predicts
scorelines and every betting market that falls out of them, for the world's
strongest leagues **and** the Tanzanian NBC Premier League, plus the UEFA and
CAF Champions Leagues.

The product owner is based in Dar es Salaam. Tanzania is a first-class market,
not an afterthought.

### The rule that governs everything

**Never publish a number you cannot defend from data.** Specifically:

- If a club has no data, say so and do not print a probability for it.
- Never let a fuzzy name match silently substitute one club for another.
- Never train or tune on information that did not exist before kick-off.
- State plainly, in the product, that the engine does **not** beat the
  bookmakers' closing price. It matches it. Anything else is false.

A previous build refused to output a Simba v Yanga prediction for weeks because
no Tanzanian data existed, and used simulated numbers only for internal
plumbing tests. That refusal was correct. Preserve that instinct.

---

## 1. The model

### 1.1 Core

A **Dixon-Coles bivariate Poisson** model fitted **separately per league**.

```
log(home goal rate) = base + attack[home] + defence[away] + home_advantage
log(away goal rate) = base + attack[away] + defence[home]
```

- `attack` is constrained to sum to zero within a league; `defence` is free.
- A Dixon-Coles `rho` term corrects the dependence at 0-0, 1-0, 0-1 and 1-1,
  which independent Poissons get wrong. Bound it to [-0.25, 0.25].
- Each match is weighted `exp(-xi * days_ago)` so recent form counts more.
- An L2 ridge (0.02) shrinks thin-sample clubs toward the league average.
- Fit with L-BFGS-B and a **hand-derived analytic gradient**. This is not an
  optimisation nicety: it made fitting 16x faster (0.30s to 0.019s) and is what
  makes the walk-forward evaluation and the parameter sweeps affordable. Verify
  it with `scipy.optimize.check_grad` in a test.

Fit three models per league: full time, first half, and second half. Half-time
markets come from the first-half model; half-time/full-time from convolving the
two halves.

### 1.2 Shots on target

Fit a **second model on shots on target**, and blend its implied goal rate with
the goals model in log space, at equal weight.

The non-obvious part: shots **alone** are worse than goals alone (0.9840 vs
0.9826). It is the blend that pays (0.9778). Goals are the target; shots are the
steadier measure of the same strength. Total shots add nothing beyond shots on
target — do not bother.

Convert a shot rate to a goal rate with the league's time-decayed goals-per-shot
conversion. Pin `rho` to zero for shot targets: the low-score correction is a
goals phenomenon and shot counts never sit at 0-0.

Where a league has no shot data (most of the world), the blend must **fall back
to goals-only automatically**, not error.

### 1.3 Shrinkage, per league

Raw ratings are over-dispersed on totals. Split each fixture's log rates into a
**level** (how many goals) and an **edge** (who scores them):

```
level = (log_lam + log_mu) / 2      edge = (log_lam - log_mu) / 2
level = league_mean + goal_shrink * (level - league_mean)
edge  = edge_scale * edge
```

Tune `goal_shrink` and `edge_scale` **per league family**, never globally by
assumption. Measured so far:

| Scope | goal_shrink | edge_scale |
|---|---|---|
| Top-10 European leagues | 0.40 | 1.10 |
| Tanzania NBC Premier League | 0.25 | 1.10 |
| Bridged UEFA ties | 1.00 | 1.10 |
| Bridged CAF ties | 1.00 | 1.10 |

Two lessons here. First, `edge_scale` moved **above 1** once shots were blended
in — with steadier rates the model should be *sharper*, not flatter, and the
calibration table caught that before the grid search did. Second, cross-league
ties want **no** level shrinkage at all: a genuine mismatch between a strong and
a weak league really does run up scores.

### 1.4 Clubs with no record in a division

Three fallbacks, in order:

1. **Carry the rating up the promotion ladder.** If the club played a division
   you also load, shift its rating by the measured gap between the two rungs:

   | Step | Attack | Defence |
   |---|---|---|
   | Premier League / Championship | 0.608 (se 0.085) | −0.633 (se 0.135) |
   | Championship / League One | 0.336 (se 0.046) | −0.420 (se 0.046) |
   | League One / League Two | 0.206 (se 0.052) | −0.228 (se 0.065) |
   | League Two / National League | 0.275 (se 0.075) | −0.171 (se 0.088) |

   Measured from 48 clubs that actually moved. On 82 traceable newcomer
   matches, carrying beat the flat prior monotonically: 1X2 log-loss 1.1135 at
   0% carry, 1.0637 at 50%, **1.0384** at 100%.

2. **The promoted-team prior.** Measured over 45 promoted team-seasons in 11 top
   divisions: a promoted side scores **0.79x** and concedes **1.16x** the league
   average (attack −0.232, se 0.031; defence +0.150, se 0.027).

3. Otherwise refuse to price the fixture and say why.

Carried ratings must be kept in a **separate map** from real ones, so
`teams`, `knows()` and the league table stay honest about who has actually
played in the division.

### 1.5 Cross-league ties (UEFA and CAF)

Each league is fitted with its own zero-sum constraint, so an English attack
rating and an Italian one are **not the same quantity**. Matches between
countries are the only thing that ties the scales together.

```
log(home rate) = c + mean_base + att_h + def_a + s_home - s_away + adv
log(away rate) = c + mean_base + att_a + def_h + s_away - s_home
```

One offset `s` per league, held to zero sum, fitted by Poisson maximum
likelihood on real cross-league results using **as-of ratings** (fitted only on
matches before each tie).

**UEFA bridge** — 439 ties, 15 leagues. On 179 held-out ties it beat having no
bridge: log-loss 1.0162 → **0.9375**, accuracy 52.5% → 58.1%.

| League | Offset | | League | Offset |
|---|---:|---|---|---:|
| Premier League | +0.454 | | Primeira Liga | −0.114 |
| Ligue 1 | +0.382 | | Austrian Bundesliga | −0.185 |
| Bundesliga | +0.287 | | Super League Greece | −0.191 |
| La Liga | +0.243 | | Scottish Premiership | −0.202 |
| Serie A | +0.212 | | Super Lig | −0.213 |
| Eredivisie | −0.024 | | Ukrainian Premier League | −0.258 |
| Eliteserien | −0.033 | | Czech First League | −0.317 |
| Jupiler Pro League | −0.042 | | | |

European home advantage in these ties: **+0.297**.

**CAF bridge** — 199 ties. On 124 held-out ties: 0.9784 → **0.9216**, 5.8%
better. Ridge 0.05. CAF home advantage is **+0.585**, nearly double Europe's —
travel and conditions are real and large.

Selected CAF offsets: Tunisia +0.497, DR Congo +0.427, Egypt +0.340, Algeria
+0.198, Morocco +0.160, **Tanzania +0.119**, South Africa +0.110, Malawi −0.018,
Botswana −0.273.

**Federation-level clubs.** In CAF most entrants come from countries with no
loadable league. Rate such a club as an *average club of its federation*:
`attack = defence = 0`, and let the federation's offset carry its strength.
Record how many CAF ties informed that offset and **show it in the UI**. A
federation with zero ties is rated as an average entrant, which almost
certainly flatters it — label that explicitly.

**Do not add a rating-transfer slope without re-measuring.** Fitting one gave
β ≈ 0.6 on training data, but β = 1 won out of sample (0.9270 vs 0.9402).
A separate check on the 64 ties that pit a real club against a federation-level
one showed the model is **under**-confident there, not over: favourites were
expected to score 1.45 and actually scored 1.88.

### 1.6 Two-legged ties

From the second leg's score matrix and the first-leg score, report
**win the tie in 90 minutes / level on aggregate / lose in 90 minutes**. Report
"level" separately rather than resolving it — it goes to extra time and
penalties, and the engine models neither. Do not assert an away-goals rule.

### 1.7 The market prior

Where a closing price exists, **de-vig it (Shin method)** and fold it into the
forecast. This is the single largest accuracy gain available, and you must
understand exactly what it is and is not.

**Measured, over 5,948 walk-forward matches:**

| | 1X2 log-loss |
|---|---|
| Model alone | 0.9739 |
| Model blended with price at 90% | 0.9568 |
| **De-vigged closing price alone** | **0.9558** |

On a held-out half, the price alone scored 0.9638 and *every* blend was equal or
worse. **The model adds nothing to 1X2 once you have the closing price.**

And the disagreements are model error, not value:

| Disagreement (max abs. difference) | n | Model LL | Market LL | Model − Market |
|---|---:|---:|---:|---:|
| 0–3 pts | 1,649 | 0.9284 | 0.9255 | +0.003 |
| 3–6 pts | 2,004 | 0.9729 | 0.9608 | +0.012 |
| 6–10 pts | 1,408 | 0.9963 | 0.9859 | +0.010 |
| 10–15 pts | 614 | 1.0040 | 0.9648 | +0.039 |
| **15+ pts** | **273** | **1.0734** | **0.9272** | **+0.146** |

When the two name *different* favourites (452 matches), the **market's
favourite wins 44.7% and the model's 24.6%**.

Betting the model's 5%+ disagreements lost **12.2% of stakes over 3,761 bets**.

**Therefore:**
- Use the price wherever it exists; the model's job there is the markets the
  price does not cover (correct score, HT/FT, team totals, margins) and
  explanation.
- The model's real value is where **no price exists**: Tanzania, CAF, lower
  tiers, and any competition the books price thinly.
- Present disagreements as a **diagnostic of what the model cannot see**, never
  as a tip. An adaptive weight that rises with disagreement was tested and added
  0.0% — do not bother reimplementing it.

### 1.8 Every market from one matrix

Build a 13x13 scoreline probability matrix, then read **every** market off that
one matrix: 1X2, double chance, draw no bet, correct score, over/under 0.5–5.5,
team totals, BTTS, clean sheet, win to nil, odd/even, winning margin, Asian
handicap (including quarter lines, with pushes returned), European handicap,
result-plus-goals combinations; and from the half models, half-time result,
half-time totals and BTTS, HT/FT, and which half has more goals.

This is a correctness property, not a convenience: no two numbers on a card can
contradict each other. A system that predicts each market with its own model
will eventually offer a customer two bets that cannot both win. Enforce it with
tests (Asian handicap at the level line must equal draw-no-bet; winning margins
must partition the 1X2; HT/FT marginals must match the HT model).

---

## 2. Evaluation — non-negotiable

- **Walk-forward only.** Fit on matches strictly before each fixture; refit
  weekly. Every model has an `as_of` cut, and a test pins it down.
- Report **log-loss, RPS, accuracy**, plus over/under and BTTS log-loss.
- Always report the **bookmaker baseline** beside your own number.
- Print **calibration tables** (predicted vs realised by probability bucket).
  These caught two real errors that aggregate metrics missed.
- Report **flat-stake and Kelly ROI** on the model's disagreements, and expect
  it to be negative.
- No feature ships without improving out-of-sample log-loss. **Rest days and
  fixture congestion were measured and rejected**: raw, short rest looks like
  1.17x more goals (p=0.001), but that is confounding — congested lists belong
  to clubs in Europe. Against the model's own expected goals as an offset, every
  coefficient collapses (p ≥ 0.29).

---

## 3. Making it more powerful — the roadmap

Ordered by expected gain per unit of effort. Items 1–3 are the only route to
genuine disagreement value.

1. **Confirmed line-ups at T−60.** The largest single lever, and the main thing
   the closing price knows that the model does not. Needs player-level attack
   and defence contributions so a named XI can be scored against the club's
   season-average XI.
2. **Availability before line-ups** — injuries, suspensions, a probable-XI model
   averaged over uncertainty, so there is a real answer three days out.
3. **Team news extracted from press conferences.** Use an LLM to pull
   **facts** (who is out, who is rested) into structured fields — not sentiment.
   Pundit opinion is weak and already in the price; log it, score it publicly,
   and do not weight it until the harness says otherwise.
4. **Opening-to-closing line movement** as a feature — it encodes when sharp
   money arrived.
5. **Shot quality (xG) rather than shot counts**, once a feed provides it.
6. **State-space ratings** (Kalman / time-varying) instead of a single
   exponential decay, so a club's rating can jump on a manager change or a
   squad overhaul rather than drifting.
7. **Hierarchical partial pooling across leagues** so thin leagues borrow
   strength from the global prior instead of relying on a fixed ridge.
8. **Stakes and motivation** computed from the table and remaining fixtures —
   title race, relegation, dead rubber, cup final three days away.
9. **Referee** (cards and penalties markets mainly) — present in the raw files
   and currently discarded.
10. **Ensemble**: Dixon-Coles as the base rate, gradient boosting on the
    residuals, market as prior. Only after 1–3.

---

## 4. League coverage — build to this list

Group A is already built. Add groups in order; each new league also strengthens
the bridges.

**A. Built (19 + 11 African).** England 1–5, Spain, Italy, Germany, France,
Netherlands, Portugal, Belgium, Turkey, Greece, Scotland, Austria, Norway,
Czechia, Ukraine; Tanzania, Egypt, Morocco, Algeria, South Africa, Nigeria,
Ghana, Kenya, Uganda, Zambia, Rwanda; UEFA and CAF Champions Leagues.

**B. Next, Europe.** Denmark, Switzerland, Poland — the strongest European
nations still missing, all with clubs in UEFA competition every season. Then
Croatia, Serbia, Sweden, Romania, Israel, Cyprus, Slovakia, Slovenia, Hungary,
Bulgaria, Finland, Ireland. Also the second tiers already in the fixtures feed:
Serie B, Segunda, 2. Bundesliga, Ligue 2, Scottish Championship/League One/Two.

**C. Next, CAF — highest value for the home market.** **Tunisia and DR Congo
first**: they are the two strongest federations in the African bridge (+0.497,
+0.427) but have no league loaded, so Espérance and TP Mazembe can only be rated
at federation level. Then Sudan, Angola, Ivory Coast, Senegal, Mali, Zimbabwe,
Mozambique, Botswana, Malawi. Add the **CAF Confederation Cup** and Tanzania's
domestic cup.

**D. Americas.** Brazil Série A and B, Argentina Primera, Liga MX, MLS,
Colombia, Chile, Uruguay, Ecuador, Paraguay, Peru. Copa Libertadores and
Sudamericana need their own continental bridge, built exactly like UEFA's.

**E. Asia and Oceania.** Saudi Pro League, Japan J1–J2, South Korea K1, Qatar,
UAE, Australia A-League, China, Iran, Uzbekistan, India ISL, Thailand, Vietnam,
Indonesia. AFC Champions League Elite as the bridge.

**F. Cups and internationals.** FA Cup, EFL Cup, Copa del Rey, DFB-Pokal, Coppa
Italia, Coupe de France. Internationals (World Cup, AFCON, Euros, Copa América,
Nations League, qualifiers) need a **separate national-team model** — do not
reuse club ratings.

**Coverage tiers, shown in the product, never hidden:**

| Tier | Data | Behaviour |
|---|---|---|
| Full | results + shots + prices | everything on; tightest intervals |
| Priced | results + prices | shots blend off, price anchors |
| Model only | results | model unaided — Tanzania, most of Africa and Asia |
| Bridged | cross-competition | UEFA/CAF bridges; federation level where needed |

---

## 5. Data layer

### 5.1 Sources

- **football-data.co.uk** — Europe, results + shots + odds. Free for personal
  use only; **a commercial product needs a licensed feed**. Note the fixtures
  feed lives at `https://football-data.co.uk/fixtures.csv` (no `www`); the
  `www` host returned 503 for a week and looked like an outage.
- **openfootball** (public domain, GitHub) — `football.txt` files for Austria,
  the `europe` repo (Norway, Czechia, Ukraine, Denmark, Switzerland, Poland,
  Croatia, Serbia, …), `world/africa` (Tanzania, Egypt, Morocco, Algeria, South
  Africa, Nigeria, Ghana, Kenya, Uganda, Zambia, Rwanda), plus UEFA and CAF
  competition archives. Results and half-time scores, no shots, no odds.
- **ligikuu.co.tz** — the official Tanzanian league site, and the only source
  that reliably carries current NBC Premier League fixtures and results. Search
  engines and the big aggregators return the *previous* season for Tanzania and
  will mislead you. Go to the official site.
- Commercial feeds for line-ups and injuries (API-Football, SportMonks,
  Sportradar). **Verify Tanzanian coverage and history depth before signing**:
  the model needs roughly a season and a half before ratings settle.

### 5.2 Source adapters

One interface per source, returning a normalised table:
`Div, Date, HomeTeam, AwayTeam, FTHG, FTAG` plus optional `HTHG, HTAG`, shots
and odds. A CSV download, a text archive and a REST API must all look identical
to everything above. This is what turns the coverage table into configuration
rather than a rewrite.

### 5.3 Pitfalls that have already bitten — handle all of these

- **Two file layouts in one source.** Austria uses `Home 1-1 (0-1) Away` in some
  seasons and `Home v Away 2-3` in others. Support both.
- **Goalscorer continuation lines** in parentheses under a result must be
  skipped, or they parse as fixtures.
- **Trailing annotations** like `[awarded]` broke a score-last pattern, and the
  fallback then swallowed the separator and invented a club called
  "FK Oleksandriya v Shakhtar Donetsk". Strip annotations, and reject any club
  name containing the separator.
- **Split identities.** The same club appears as `LASK` and `LASK Linz`,
  `RB Salzburg` and `FC Salzburg`. Left alone one club becomes two half-rated
  ones. Canonicalise by stripping club-type tokens, accents and hyphens — and
  **report every merge** so a human can eyeball it. It made exactly three on
  Austria, all correct.
- **Season year inference.** Derive the year from the season in the filename and
  the month, never by incrementing on a month decrease — multi-stage files
  restart at September and will run your dates into 2030.
- **Calendar-year seasons** (Scandinavia) have a single year in the filename.
- **Column filters silently dropping data.** A fixtures loader that keeps a
  fixed column list will drop first-leg scores and schedule notes, and the
  feature that depends on them fails quietly. Keep optional columns explicitly.
- **Domestic second tiers are not cup ties.** Routing an unloaded domestic
  division through the cross-league path produces one bogus "missing club" note
  per fixture, and can misfile a second-tier match as a cross-league tie.

### 5.4 Name resolution — the highest-risk component

A loose fuzzy match silently predicts a *different club* and looks confident.
This is the worst failure the product can have.

- Batch prediction does **no fuzzy matching at all**.
- Interactive lookup may fuzzy match, but only with a tight cutoff (0.85) and a
  margin over the runner-up — and it must **report the interpretation**
  ("Read 'Man Utd' as 'Man United'").
- Similarity alone cannot do this job: `Sheff Utd`/`Sheffield United` scores
  **0.72**, *below* `Le Mans`/`Lens` at **0.73**. So common abbreviations go
  through an explicit alias table, and fuzzy stays tight enough to reject both.
- Fold Nordic and Slavic characters (ø, å, š, ž, ř, ı, ş, ğ) on both the query
  and the candidate pool.
- Regression tests must assert that `Le Mans` never resolves to `Lens`,
  `Amedspor` never to `Pendikspor`, `Erzurumspor` never to `Bodrumspor`.

### 5.5 Point-in-time correctness

Every row carries `known_at`. The store must be able to answer "what did we know
at kick-off minus 60 minutes". Temporal leakage is the failure mode that makes a
backtest report a fantasy you then build on confidently. Enforce it in the
schema, not by discipline.

---

## 6. The apps

### 6.1 Stack

- **Model service** — Python + FastAPI. Returns a score matrix; the API turns it
  into markets.
- **Application API** — Spring Boot. Accounts, subscriptions, favourites,
  alerts, the published record.
- **Web** — React. **Mobile** — React Native, sharing card components and
  formatting with web.
- **Postgres** for matches, features and the published record.

### 6.2 Tanzania-specific product decisions

- **Ship a PWA before a store app.** Store installs cost data and friction on a
  mid-range Android. A PWA installs from a link, updates without a download and
  caches the day's slate offline. Budget **under 150KB first load** as a hard
  limit — the content is text and tables, so it is achievable.
- **WhatsApp is a first-class channel**, not a afterthought. Send the day's
  slate; let someone reply with a fixture name and get the card back. Many users
  will never install anything.
- **Mobile money, not cards**: M-Pesa, Mixx by Yas, Airtel Money, Halopesa, via
  an aggregator (Selcom, Pesapal, ClickPesa, Flutterwave). Confirm payments on
  **aggregator webhooks, never by parsing SMS**. Price in TZS, weekly tier
  first — monthly commitments convert badly.
- **Swahili and English from the first release.** The market names are the hard
  part (*timu zote mbili kufunga* for both teams to score); settle them with
  people who actually bet, before launch.
- **Times in East Africa Time**, converted properly from the feed's UK times via
  a real time-zone database — a late UK kick-off lands on the next calendar day
  in Dar es Salaam. Show the date on **every** row, not just the time.

### 6.3 Screens

1. **Today** — every fixture in the next 24 hours, grouped by competition, with
   the Tanzanian league and CAF pinned above Europe by default. One row per
   match: kick-off, 1/X/2, pick, over 2.5, BTTS, and the tier badge. Fastest and
   lightest screen in the app.
2. **Match card** — every market off one matrix; expected goals, correct scores,
   handicaps, half-time. Where a price exists, show the model's own view beside
   the market's rather than merging them away.
3. **Club** — attack/defence ratings, recent results, rating movement. For Simba
   and Yanga, CAF form beside domestic form.
4. **Record** — every prediction published before kick-off, immutable, with
   realised accuracy and calibration by competition. This is the one claim in
   this market that cannot be faked, and almost nobody makes it.
5. **Model vs market** — where they differ most, labelled as a diagnostic.

### 6.4 Regulation

A prediction product sold in Tanzania sits near the Gaming Board of Tanzania's
advertising and responsible-gambling rules. No guaranteed returns, visible 18+
marking, deposit-limit and self-exclusion signposting. Take local licensing
advice before charging money. Given the measured −12.2%, **the honest claim and
the compliant claim are the same one**.

---

## 7. Build order

1. Port the engine as specified in §1, with the evaluation harness of §2 from
   day one. Reproduce the measured constants as regression tests.
2. Source adapters and the competition registry, with `known_at` in the schema.
3. Per-competition parameter sweeps. Never reuse Europe's constants for a new
   league family without measuring.
4. The published record, before any paying users.
5. PWA (Today + Match card), then WhatsApp delivery, then payments.
6. Line-ups and availability — the only route to genuine edge.

## 8. Acceptance criteria

- Walk-forward reproduces: 1X2 log-loss ≈ 0.974 model-alone, ≈ 0.957 blended,
  bookmaker baseline ≈ 0.956, over ~5,900 top-10 European matches.
- Bridges reproduce: UEFA 1.0162 → 0.9375 on held-out ties; CAF 0.9784 → 0.9216.
- Calibration tables track within a couple of points per bucket.
- Market-consistency tests pass (§1.8).
- Name-resolution regression tests pass (§5.4).
- No fixture is ever dropped silently: anything unpredictable appears in the
  output with its reason.
