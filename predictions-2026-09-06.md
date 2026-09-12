# Football predictions

**Sunday 06 September 2026, kick-offs from 13:30**

Generated Sun 06 Sep 2026, 15:34 &middot; Dixon-Coles bivariate Poisson fitted on 14,788 historical matches.
Where a closing price exists, **90%** of the forecast is taken from it and 10% from the model.

> This is analysis, not tipping. The engine is well calibrated but does **not** beat the closing line: backtested over 5,948 matches it scores 0.9739 log-loss against the market's 0.9569, and betting its disagreements lost 12.2% on flat stakes. Percentages here mean what they say; they are not an edge.

## No fixtures in this window

The fixtures feed on disk covers **2026-08-28 to 2026-08-31** and holds nothing for this window. The live feed at football-data.co.uk is returning HTTP 503 right now, so it could not be refreshed.

That is very likely correct rather than a gap in the data. Across the same three days of previous seasons, these ten leagues played:

| Season | Matches on these dates |
|---|---:|
| 2023 | 0 |
| 2024 | 0 |
| 2025 | 1 |
| 2026 | 0 |

Early September is the FIFA international window and Europe's domestic leagues pause through it. This engine rates club sides only — it has no international-team model — so there is nothing today it can honestly forecast.

Club football in these leagues resumes the following weekend. Re-run this once the feed is back:

```bash
python predict.py refresh-fixtures && \
  python predict.py brief --top10 -o slate.md
```

---

## What the engine currently rates

Nothing to forecast today does not mean nothing is known. These are the live ratings every prediction is built from, as of this run. **Attack** is goals scored relative to the league average, **defence** is goals conceded — so lower defence is better — and **rating** is the ratio of the two.

### Premier League (England)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Arsenal | 1.49 | 0.46 | **3.22** |
| 2 | Man City | 1.53 | 0.59 | **2.59** |
| 3 | Liverpool | 1.46 | 0.73 | **2.00** |
| 4 | Chelsea | 1.32 | 0.75 | **1.75** |
| 5 | Newcastle | 1.30 | 0.84 | **1.54** |
| 6 | Aston Villa | 1.15 | 0.76 | **1.51** |

### La Liga (Spain)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Barcelona | 2.12 | 0.61 | **3.46** |
| 2 | Real Madrid | 1.69 | 0.60 | **2.84** |
| 3 | Ath Madrid | 1.38 | 0.56 | **2.45** |
| 4 | Villarreal | 1.51 | 0.74 | **2.05** |
| 5 | Betis | 1.22 | 0.78 | **1.57** |
| 6 | Celta | 1.26 | 0.82 | **1.54** |

### Serie A (Italy)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Inter | 1.90 | 0.62 | **3.08** |
| 2 | Atalanta | 1.40 | 0.65 | **2.15** |
| 3 | Napoli | 1.33 | 0.62 | **2.14** |
| 4 | Juventus | 1.45 | 0.68 | **2.11** |
| 5 | Milan | 1.37 | 0.65 | **2.11** |
| 6 | Como | 1.40 | 0.69 | **2.03** |

### Bundesliga (Germany)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Bayern Munich | 2.20 | 0.56 | **3.92** |
| 2 | Dortmund | 1.45 | 0.67 | **2.18** |
| 3 | Leverkusen | 1.39 | 0.75 | **1.86** |
| 4 | RB Leipzig | 1.22 | 0.74 | **1.65** |
| 5 | Stuttgart | 1.33 | 0.81 | **1.63** |
| 6 | Ein Frankfurt | 1.27 | 0.91 | **1.39** |

### Ligue 1 (France)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Paris SG | 1.84 | 0.55 | **3.37** |
| 2 | Lens | 1.26 | 0.62 | **2.05** |
| 3 | Marseille | 1.59 | 0.79 | **2.01** |
| 4 | Monaco | 1.37 | 0.73 | **1.86** |
| 5 | Lyon | 1.24 | 0.67 | **1.84** |
| 6 | Lille | 1.17 | 0.64 | **1.83** |

### Primeira Liga (Portugal)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Sp Lisbon | 2.19 | 0.45 | **4.92** |
| 2 | Benfica | 1.86 | 0.46 | **4.07** |
| 3 | Porto | 1.60 | 0.40 | **3.96** |
| 4 | Sp Braga | 1.58 | 0.62 | **2.54** |
| 5 | Famalicao | 1.04 | 0.60 | **1.72** |
| 6 | Gil Vicente | 1.08 | 0.72 | **1.50** |

### Eredivisie (Netherlands)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | PSV Eindhoven | 2.06 | 0.64 | **3.23** |
| 2 | Feyenoord | 1.57 | 0.61 | **2.55** |
| 3 | Ajax | 1.38 | 0.64 | **2.15** |
| 4 | Twente | 1.27 | 0.61 | **2.09** |
| 5 | Nijmegen | 1.46 | 0.77 | **1.91** |
| 6 | Utrecht | 1.15 | 0.62 | **1.87** |

### Jupiler Pro League (Belgium)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | St. Gilloise | 1.38 | 0.41 | **3.35** |
| 2 | Club Brugge | 1.69 | 0.67 | **2.52** |
| 3 | Anderlecht | 1.24 | 0.70 | **1.76** |
| 4 | Genk | 1.30 | 0.80 | **1.62** |
| 5 | Antwerp | 0.99 | 0.69 | **1.43** |
| 6 | Gent | 1.25 | 0.90 | **1.39** |

### Super Lig (Turkey)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Galatasaray | 1.93 | 0.46 | **4.21** |
| 2 | Fenerbahce | 1.94 | 0.62 | **3.14** |
| 3 | Trabzonspor | 1.49 | 0.69 | **2.17** |
| 4 | Besiktas | 1.34 | 0.63 | **2.14** |
| 5 | Goztep | 1.13 | 0.61 | **1.85** |
| 6 | Buyuksehyr | 1.35 | 0.75 | **1.78** |

### Super League Greece (Greece)

| # | Club | Attack | Defence | Rating |
|---:|---|---:|---:|---:|
| 1 | Olympiakos | 1.52 | 0.39 | **3.88** |
| 2 | AEK | 1.59 | 0.49 | **3.25** |
| 3 | PAOK | 1.77 | 0.56 | **3.15** |
| 4 | Panathinaikos | 1.46 | 0.62 | **2.37** |
| 5 | Levadeiakos | 1.39 | 0.89 | **1.56** |
| 6 | Aris | 0.90 | 0.66 | **1.36** |

Name any two clubs and a card can be produced immediately — the engine does not need a fixture list to price a match:

```bash
python predict.py predict "Arsenal" "Chelsea" --full --best
```
