# Deploying

Three processes: Postgres, the API, and a static web server.

```bash
cp .env.example .env        # set LIVE_API_KEY, POSTGRES_PASSWORD, SOCCER_DATA_DIR
docker compose up --build   # web on :8080, API on :8000
```

Then check the one thing that fails silently:

```bash
curl http://localhost:8000/api/health
```

`data.missing_top10` must be an empty list. If it names E0, SP1, I1 and the rest,
the European pool is not mounted — see *Match data* below.

Production is the same shape with real secrets and TLS in front. TLS is not
optional: the phone app is served from `https://localhost` inside its WebView
and Android blocks cleartext, so a plain-http API gives you an app where every
screen is empty.

## Where the API-Football key goes

On the **service**, as `LIVE_API_KEY`. Nowhere else.

- **Docker** — set it in `.env` beside `docker-compose.yml`. Compose
  substitutes it into the `api` container only; the `web` container never sees
  it. `.env` is git-ignored.
- **Local run, PowerShell** — `$env:LIVE_API_KEY = "your-key"`, then start
  uvicorn in the same window.
- **Local run, bash** — `LIVE_API_KEY=your-key .venv/Scripts/python.exe -m uvicorn service.app:app --port 8000`

**Never** put it in `web/.env` or in any variable starting with `VITE_`. Vite
inlines those into the JavaScript bundle, which every browser downloads and
every copy of the phone app carries. A test
(`test_the_web_app_has_no_route_to_the_provider`) fails if the provider's host
or key header ever appears in frontend code.

Check it took, without spending a request:

```bash
python predict.py live-status
```

## How live scores stay inside the free tier

The free tier is **100 requests a day and 10 a minute**, and exceeding it can
get the key or the server's IP temporarily blocked. The previous code polled 38
leagues one request each, on the user's request path: a single cold page load
was 38 calls, and two of them spent the day.

```
Frontend  ─►  /api/live  ─►  database snapshot
                                  ▲
               one refresher, only while a match we carry is in play
                                  │
                            API-Football
```

1. **The frontend never reaches the provider.** `/api/live` reads the last
   snapshot, however many clients poll it.
2. **No call unless something is in play.** Our schedule knows every kick-off;
   outside kick-off −5 min to +130 min, nothing is spent — most of every day.
3. **One request per poll** — `fixtures?live=all`, not one per league.
4. **Every attempt is charged before it is sent**, against a rolling 24-hour
   window with 10 requests held in reserve. A 429, or a 200 whose `errors`
   mention a limit, stops polling for an hour instead of retrying into a block.
   The provider's own remaining-quota header is trusted over our count.

The honest trade-off: on the free tier, live scores update every **3 to 7
minutes**, not in real time. The refresher spreads what is left of the budget
across what is left of the day's play — a quiet evening polls every 3 minutes,
a full Saturday stretches to about 7 so the budget lasts until the last final
whistle. Real-time scores need a paid plan; raise `LIVE_DAILY_LIMIT` to match.

To find a league's API-Football id (spends one request):

```bash
python predict.py live-leagues tanzania
```

## Match data

Two sources, shipped two ways:

| Data | How it gets into the container |
|---|---|
| African, Nordic, Eastern European divisions; Tanzanian overlays | baked into the image from `data/` |
| Every European division + the fixtures feed (football-data.co.uk) | **mounted** at `/data/soccer` from `SOCCER_DATA_DIR` |

The European pool lives outside the repo and is refreshed in place, so it
cannot be baked in. Point `SOCCER_DATA_DIR` at the folder you already have, or
leave it unset and download a fresh copy into `./soccerdata`:

```bash
docker compose exec api python predict.py update --since 2023
```

A missing mount does not fail — the loader serves the bundled divisions alone,
fifteen instead of thirty, and every endpoint still answers. That is why
`/api/health` reports `data.missing_top10`.

## Storage: what lives where

| Data | Where | Why |
|---|---|---|
| Match results and fixtures | files (bundle in image, European pool mounted) | read-only; the engine re-fits from it at boot |
| Published record | Postgres, or the `RECORD_ROOT` volume | cannot be regenerated if lost |
| Live-score ledger + snapshot | Postgres, or SQLite beside the record | the budget must survive restarts |
| Accounts, payments, uploads | Postgres | concurrent writes; money needs transactions |

Tables are created on boot when `DATABASE_URL` is set. Without it, everything
falls back to files — fine for one machine and **one worker**. More than one
worker needs Postgres: two processes appending to the record's CSV corrupt its
hash chain, and without the advisory lock two refreshers can race for the same
live-score request.

### Switching an existing record to Postgres

Turning on `DATABASE_URL` alone starts an **empty** record: publishing restarts
the hash chain from zero and everything already published is orphaned. Migrate
once, straight after the switch:

```bash
docker compose exec api python predict.py record-migrate
```

Rows cross verbatim — original timestamps, `prev` and `hash` — and the chain is
verified on both sides. It refuses a file whose own chain is broken, and a
database holding rows the file does not; it never splices two histories.
Re-running is harmless and appends anything published to the file since.

## Health

`GET /api/health` answers 200 whenever the process is up, and reports what a
deploy needs to know: whether the eleven-second model warm-up has finished,
which match data loaded, whether the database is reachable, and the live-score
budget. It deliberately does not fail during warm-up, or an orchestrator would
restart the service in a loop while the models fit.

## Publishing the record

The record only counts predictions written before kick-off, and cannot be
backfilled — a missed day is gone. Schedule it:

```bash
python predict.py record-publish     # or POST /api/record/publish
```

Mid-morning and late afternoon EAT covers both European and Tanzanian
kick-offs. It is idempotent, so running it more often costs nothing.
