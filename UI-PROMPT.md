# Design & product prompt — football predictor web + mobile UI

Paste this into the agent building the interface. It assumes the prediction
engine described in `BUILD-PROMPT.md` already exists and is reachable over an
API. This document is about what people see, touch and pay for.

---

## 0. Read this first — the two rules everything else bends around

**Rule 1: the product sells understanding, not tips.** The engine matches the
bookmakers' closing price; it does not beat it (measured: price alone 0.9558
log-loss, best model blend 0.9568, and betting the model's disagreements lost
12.2% of stakes over 3,761 bets). So the interface must never look like a
tipster app. No "BANKER OF THE DAY", no flame icons, no green "GUARANTEED"
badges, no accumulator builder that implies edge. That styling is both dishonest
here and a regulatory problem in Tanzania.

What you sell instead: *why* Arsenal are 61%, which scorelines carry it, what
moved when the line-up dropped, and a public record proving the percentages mean
what they say.

**Rule 2: never present a derived pairing as a scheduled fixture.** The engine
can price any two clubs without a schedule — it also fills gaps by listing every
unplayed pairing in a league. Those are *ratings of a matchup*, not fixtures.
A current build shipped 614 "fixtures" of which roughly 110 were real; the rest
were round-robin fallbacks shown with an empty kick-off column, in leagues where
the real schedule simply had not been fetched.

Enforce this in the UI:

- A fixture **must** have a date and kick-off time to appear under "Fixtures".
- Priced pairings live in a separate, clearly labelled surface
  ("Head-to-head", "Rate any matchup"), never mixed into a date-ordered slate.
- The fixture count in the header counts fixtures only.
- If a league has no schedule loaded, say so in that league's section
  ("No schedule loaded for the Tanzanian Premier League — showing head-to-head
  ratings instead") rather than silently substituting.

---

## 1. Visual direction

Do **not** produce the default AI-app look (purple-blue gradient hero, Inter
everywhere, rounded cards with a left accent bar, emoji section markers). Do not
produce the default betting-app look either (black + acid green, odds in
pill-shaped buttons).

Ground the identity in football's **printed heritage**: the matchday programme,
the pools coupon, the newspaper results grid — dense, confident, numerate — with
a modern data layer on top. Build two directions and pick one:

**Direction A — "Matchday programme".** Warm off-white paper ground, deep ink,
one strong spot colour used the way a club programme uses it. Condensed
grotesque for scorelines and team names, a readable humanist face for prose,
tabular monospace for every number that sits in a column. Hairline rules rather
than boxes. Feels editorial and trustworthy.

**Direction B — "Floodlit night".** Deep blue-black ground, cool slate
neutrals, a single warm signal colour (amber/ochre) reserved *only* for
model-versus-market divergence, plus separate semantic colours for good/
warning/critical. Feels like a broadcast stats overlay.

Whichever you pick:

- Neutrals must be hue-biased toward the accent, never pure grey.
- Pitch green may appear as a semantic accent; it must not be the whole brand.
- Support light and dark properly: define the complete palette as tokens on
  `:root`, redefine only tokens under `prefers-color-scheme: dark` and under an
  explicit theme attribute. Never define a colour solely inside a media query.
- `font-variant-numeric: tabular-nums` on every column of digits, without
  exception. Probabilities that jitter as they change look broken.
- Give percentages a consistent visual weight: 61% and 6% must be scannable in
  the same column without one shouting.

---

## 2. Information architecture

Six surfaces. Nothing else at the top level.

1. **Today** — the default screen. Every *real* fixture in the next 24–48 hours,
   grouped by competition, Tanzanian league and CAF pinned above Europe by
   default (user-reorderable). One row: kick-off (in the user's time zone),
   match, 1/X/2, pick, over 2.5, BTTS, and a data-tier badge.
2. **Match** — the full card. Every market off one scoreline distribution.
3. **Club** — attack/defence ratings, recent form, rating movement over the
   season; for Simba and Yanga, CAF form beside domestic form.
4. **My fixtures** — upload or build your own list (§4).
5. **Record** — every prediction published before kick-off, immutable, with
   realised accuracy and calibration by competition.
6. **Model vs market** — where the two differ most, framed as a diagnostic.

### Long lists

600+ rows must never render at once. Virtualise the list, paginate server-side,
and give the Today screen: competition filter, date chips, a confidence filter,
and a search box. Default to "today only" and let people widen it.

---

## 3. The match card — the centrepiece

This is the screen the product lives or dies on. It must feel like opening a
team sheet, not reading a spreadsheet.

**Above the fold:** the two clubs, kick-off in local time, the 1/X/2 split as a
single horizontal stacked bar with the three numbers on it, expected goals, and
the most likely scoreline. Plus a plain-language confidence phrase ("Clear",
"Slight lean", "Close to a coin toss") — never a bare percentage alone.

**The signature graphic: the scoreline grid.** A 6×6 heatmap of the score
matrix, home goals down, away goals across, intensity by probability, the modal
cell called out. Nothing else in this category shows this, it is genuinely what
the engine computes, and it makes the "most likely score ≠ most likely result"
point visually in one glance. Make it tappable: tapping a cell shows that exact
scoreline's probability and what it implies.

**Then, progressively disclosed sections:** double chance, totals ladder
(0.5–5.5 as a small multiples strip), both teams to score, team totals, clean
sheet, Asian and European handicaps, winning margin, half-time, half-time/full
time, result+goals combinations.

**Playful, honest touches** (each must encode something true):

- **Neutral-venue toggle.** The engine supports it. Flip it and watch the bar
  move — home advantage becomes tangible rather than asserted.
- **Two-leg tie widget** for CAF/UEFA knockouts: first-leg score carried in,
  showing "through in 90 / level on aggregate / out in 90". Level stays
  unresolved, because extra time and penalties are not modelled.
- **"What the model can't see"** — a small, permanent panel listing injuries,
  line-ups, transfers and congestion as *known blind spots*. Turn the weakness
  into a trust signal.
- **Rating movement sparkline** per club.
- **Share as image.** One tap produces a clean card image sized for WhatsApp.
  In Tanzania this is the single highest-leverage growth feature in the app.
- Micro-interactions only where they carry meaning: bars animating to their
  value on load, the grid cells settling in a quick stagger. Respect
  `prefers-reduced-motion`. No confetti, no spinning balls.

**Data provenance, always visible.** Every card states what fed it: a tier badge
(Full / Priced / Model-only / Bridged) and, where relevant, a line such as
"Gaborone United has no league data — rated as an average Botswanan CAF entrant
from 6 continental matches". Never hide thin evidence behind a confident number.

---

## 4. "My fixtures" — user-uploaded lists

The feature the owner asked for, and the one with the most ways to go wrong.

**Input methods**, in build order:
1. Paste text — one fixture per line, `Home v Away`, optional date/time.
2. Upload CSV or XLSX, with a column-mapping step.
3. Pick from the schedule (search and add).
4. Later: photograph a printed coupon (OCR) — do not attempt until 1–3 are solid.

**The name-matching step is the product's biggest risk.** A loose match silently
predicts a *different club* and looks confident doing it. The engine already
refuses fuzzy matching in batch mode; the UI must make its decisions visible:

- Show a **reconciliation table** before any prediction runs: each uploaded name,
  what it matched, which competition, and a status — exact / interpreted /
  ambiguous / not found.
- **Interpreted** matches (e.g. "Man Utd" → "Man United") are shown plainly and
  are accepted by default but individually overridable.
- **Ambiguous** rows block until the user picks from a short list. Never guess.
- **Not found** rows stay in the table with a clear reason ("no data for this
  club in any loaded competition") and are excluded from the results, never
  dropped silently.
- Let the user set the competition per row when it is ambiguous.
- Remember corrections per user, so the second upload is cleaner than the first.

**Output:** the same card grid as Today, plus export to PDF, CSV and a share
image. Show the tier badge per row so a user can see which of their fixtures the
model knows well.

**Guard rails:** free tier limited to a small number of rows per day; premium
unlimited within reason. Validate file size and row count before upload. Show a
progress state — a 200-row list will take a moment.

---

## 5. Monetisation

Price for Tanzania first. Card penetration is low, mobile money is universal,
and monthly commitments convert badly — lead with a weekly tier in TZS.

### Free

- Today's fixtures for a rotating set of competitions, including the full
  Tanzanian Premier League and CAF (this is the hook — nobody else covers them
  well).
- 1X2, over/under 2.5, both teams to score.
- The public record, in full. Never paywall the evidence that the product is
  honest.
- Three uploaded fixtures per day.

### Premium

- Every market on every card: correct score, half-time/full-time, Asian and
  European handicaps, team totals, winning margin, combinations.
- All competitions, including the deep European and African coverage.
- Unlimited uploads and exports (PDF/CSV/image).
- Alerts: line-ups dropped, rating moved, your club plays in an hour.
- Model-versus-market view with the divergence history.
- Ad-free.

### Pricing and payment

- Weekly and monthly tiers in TZS; anchor on the weekly.
- Mobile money via an aggregator (Selcom, Pesapal, ClickPesa, Flutterwave) —
  M-Pesa, Mixx by Yas, Airtel Money, Halopesa. Confirm on **aggregator
  webhooks, never by parsing SMS**.
- Subscribe and renew **inside WhatsApp** as well as in-app. Many users will
  never open an app store.
- Free trial of a few days, and a referral that gives both sides a free week.
- Grace period on failed renewal, and a clear, one-tap cancel. Trust is the
  product; dark patterns poison it.

### Other revenue, later

- **API / B2B**: media houses and betting affiliates paying for the feed.
- **White-label** match cards for a sports publisher.
- Sponsorship of a competition section, clearly marked as advertising.
- Do **not** take affiliate commission on bet placement in a way that biases
  what you show. If you ever do, disclose it on every affected screen.

### Hard constraints

18+ marking, no guaranteed-return language anywhere, deposit-limit and
self-exclusion signposting, and responsible-gambling copy that is not buried.
Take Gaming Board of Tanzania advice before charging. The measured −12.2% means
the honest claim and the compliant claim are the same one.

---

## 6. Technical constraints

- **PWA first**, then React Native. Store installs cost data and friction on a
  mid-range Android.
- **First load under 150KB** — a hard budget, not an aspiration. The content is
  text and tables. Code-split the match card; lazy-load the heatmap.
- **Offline**: cache today's slate and the last viewed cards. The app must open
  and show something useful on a dropped connection.
- **Swahili and English from the first release.** The market names are the hard
  part (*timu zote mbili kufunga* for both teams to score) — settle them with
  people who actually bet, before launch, not with a translation API.
- **Times in the user's zone**, converted from the feed's UK times with a real
  time-zone database. A late UK kick-off lands on the next calendar day in Dar
  es Salaam. Show the **date on every row**, not just the time — a multi-day
  list without dates is unreadable.
- Accessibility: visible keyboard focus, AA contrast in both themes, the
  scoreline grid needs a text alternative, and colour must never be the only
  carrier of meaning (pair every colour cue with a label or shape).
- Wide tables scroll inside their own container; the page body never scrolls
  sideways.

---

## 7. Acceptance criteria

- No row appears under "Fixtures" without a date and kick-off time.
- The header count matches the number of real fixtures shown.
- An uploaded list containing "Le Mans", "Man Utd" and "Notaclub FC" produces:
  one interpreted match shown to the user, one exact, one explicit not-found —
  and never silently substitutes a different club.
- A 600-row list renders smoothly on a mid-range Android.
- Every card shows its data tier and any federation-level or promoted-team
  fallback in plain language.
- First load under 150KB; the app opens offline with cached content.
- The whole interface works in Swahili, including market names.
- Nowhere in the product does any copy claim an edge over bookmakers.
