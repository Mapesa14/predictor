"""The mutable half of the system.

The split this module exists to enforce: **match data stays on disk, user data
goes in the database.**

Results and fixtures are read-only reference data. The engine reads the whole
pool into memory at boot and re-fits from scratch, so a database there would
only be a slower file - measured: 26,199 matches, 8 MB, and the slate's cost
was never storage but repeated in-process work.

What genuinely needs a database is everything that is written while the service
is running, by more than one process, where losing a write matters:

  * the published record - append-only and hash-chained, which a file handles
    fine on one box and not at all across two. Two workers appending to the
    same file corrupt the chain.
  * accounts and subscriptions - money, so nothing less than a transaction.
  * fixture lists people upload.

Backends are chosen by `DATABASE_URL`. Unset means the file store, which is
what a single machine should keep using: no server to run, and the record is a
CSV you can read in any text editor. Set it and the same code writes to
Postgres instead.

The tests run against SQLite, which is what makes the SQL here honest - it is
the same SQLAlchemy Core statements on both, not a parallel implementation.
Postgres-specific behaviour (advisory locks for the chain, `SERIALIZABLE`) is
called out where it applies.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData,
                        String, Table, Text, UniqueConstraint, create_engine,
                        select)

metadata = MetaData()

# ---------------------------------------------------------------- the record
# Mirrors predictor/record.py's CSV exactly, including `prev` and `hash`, so
# either store holds the same rows and the chain verifies the same way.
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("published_at", String(40), nullable=False),
    Column("div", String(16), nullable=False),
    Column("league", String(80)),
    Column("date", String(10), nullable=False),
    Column("kickoff", String(40), nullable=False),
    Column("home", String(80), nullable=False),
    Column("away", String(80), nullable=False),
    Column("pH", Float), Column("pD", Float), Column("pA", Float),
    Column("pOver25", Float), Column("pBTTS", Float),
    Column("score", String(12)),
    Column("xgH", Float), Column("xgA", Float),
    Column("mk1", Float), Column("mkX", Float), Column("mk2", Float),
    Column("prev", String(64), nullable=False),
    Column("hash", String(64), nullable=False, unique=True),
    # One prediction per fixture, enforced by the database rather than by a
    # read-then-write in the application: two workers publishing the same slate
    # at the same moment is the ordinary case, not the rare one.
    UniqueConstraint("div", "date", "home", "away", name="uq_prediction_fixture"),
)

# ------------------------------------------------------------------ accounts
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # Phone first: this is a Tanzanian product and mobile money is the rail.
    # Email is optional and may legitimately never be set.
    Column("msisdn", String(24), unique=True),
    Column("email", String(160), unique=True),
    Column("display_name", String(80)),
    Column("created_at", DateTime, nullable=False),
    Column("locale", String(8), server_default="en"),
    Column("disabled", Boolean, server_default="0"),
)

subscriptions = Table(
    "subscriptions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False, index=True),
    Column("plan", String(24), nullable=False),        # free | weekly | monthly
    Column("status", String(24), nullable=False),      # trial|active|grace|lapsed
    Column("started_at", DateTime, nullable=False),
    Column("expires_at", DateTime),
    Column("currency", String(8), server_default="TZS"),
    Column("amount_minor", Integer),                   # cents/senti, never float
    Column("created_at", DateTime, nullable=False),
)

# Every mobile-money callback, stored raw before it is interpreted. An
# aggregator that sends the same webhook twice must not bill twice, so the
# provider's own reference is unique and settlement reads from this table
# rather than from whatever the request handler happened to parse.
payments = Table(
    "payments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, index=True),
    Column("provider", String(32), nullable=False),    # selcom, clickpesa, ...
    Column("provider_ref", String(120), nullable=False),
    Column("status", String(24), nullable=False),
    Column("currency", String(8), nullable=False),
    Column("amount_minor", Integer, nullable=False),
    Column("received_at", DateTime, nullable=False),
    Column("raw", Text),
    UniqueConstraint("provider", "provider_ref", name="uq_payment_ref"),
)

# ------------------------------------------------------- uploaded fixtures
uploads = Table(
    "uploads", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, index=True),
    Column("name", String(160)),
    Column("created_at", DateTime, nullable=False),
    Column("rows", Integer),
    Column("unresolved", Integer),      # names the matcher refused to guess at
    Column("body", Text),               # the list as given, kept verbatim
)

# ------------------------------------------------------------- live scores
# API-Football's free tier is 100 requests a day and 10 a minute, and going
# over does not just fail: it can get the key or the server's IP temporarily
# blocked. So the provider is never called on a user's request. The frontend
# reads `live_snapshot`; a single refresher decides whether a call is worth
# spending, and every attempt is written to `provider_calls` first.
#
# Both live in the database rather than in memory for the two reasons an
# in-memory counter fails: a restart forgets how much of today is spent, and
# two server processes each believe they have the whole budget.

# One row per *attempted* call, inserted before the request goes out and
# updated after. Charging on attempt is the conservative accounting - it is not
# documented whether a failed call counts against the quota, and a process
# that dies mid-request must still be billed to itself.
provider_calls = Table(
    "provider_calls", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("provider", String(32), nullable=False),
    Column("endpoint", String(120), nullable=False),
    Column("called_at", DateTime, nullable=False, index=True),
    Column("http_status", Integer),
    Column("ok", Boolean, nullable=False, server_default="0"),
    # What the provider itself says is left. Trusted over our own count when
    # present, because it also sees calls made from anywhere else with the key.
    Column("remaining_day", Integer),
    Column("remaining_minute", Integer),
    Column("note", String(240)),
)

# The latest normalised provider payload - the only thing /api/live serves.
# Old rows are kept briefly for diagnosis and pruned by the refresher.
live_snapshot = Table(
    "live_snapshot", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("provider", String(32), nullable=False),
    Column("fetched_at", DateTime, nullable=False, index=True),
    Column("matches", Integer, nullable=False),
    Column("body", Text, nullable=False),
)


# ------------------------------------------------------------------- engine
def url() -> str | None:
    """The configured database, or None for the file store."""
    return os.environ.get("DATABASE_URL") or None


CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", "5"))

_engine = None


def engine(dsn: str | None = None, echo: bool = False):
    """A process-wide engine. `dsn` is for tests; production reads the env."""
    global _engine
    target = dsn or url()
    if target is None:
        return None
    if _engine is None or str(_engine.url) != str(_normalise(target)):
        dsn = _normalise(target)
        # Fail fast on an unreachable host. Without this the default is the
        # OS-level TCP timeout - over two minutes on Windows - so a wrong
        # DATABASE_URL hangs the health check instead of answering it.
        args = {"connect_timeout": CONNECT_TIMEOUT} \
            if dsn.startswith("postgresql") else {}
        _engine = create_engine(dsn, echo=echo, future=True,
                                pool_pre_ping=True, connect_args=args)
    return _engine


def _normalise(dsn: str) -> str:
    """Accept the `postgres://` form platforms hand out.

    Heroku, Railway and Render still emit `postgres://`, which SQLAlchemy 2
    does not recognise; and the default driver for `postgresql://` is psycopg2
    while the maintained one is psycopg 3. Both rewrites are here so a copied
    connection string works rather than failing at import time.
    """
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    if dsn.startswith("postgresql://"):
        dsn = "postgresql+psycopg://" + dsn[len("postgresql://"):]
    return dsn


def create_all(eng=None) -> None:
    eng = eng or engine()
    if eng is None:
        raise RuntimeError("no DATABASE_URL set")
    metadata.create_all(eng)


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def healthy(eng=None) -> dict:
    """Whether the database is reachable and initialised - for /api/health.

    Engine construction is inside the guard, not outside it. A missing driver
    raises at `create_engine`, before any connection is attempted, so building
    the engine first turned "Postgres is configured but psycopg was never
    installed" into a stack trace on a health check rather than a status it
    could report.
    """
    if eng is None and url() is None:
        return {"configured": False, "ok": True, "backend": "files"}
    try:
        eng = eng or engine()
        with eng.connect() as c:
            n = c.execute(select(predictions.c.id).limit(1)).fetchall()
        return {"configured": True, "ok": True, "backend": eng.dialect.name,
                "has_rows": bool(n)}
    except Exception as e:                    # a dead database is not a crash
        return {"configured": True, "ok": False, "backend": "unknown",
                "error": "%s: %s" % (type(e).__name__, e)}
