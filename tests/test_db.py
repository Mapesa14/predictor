"""The database backend, on SQLite.

SQLite is the substrate, Postgres is the target. That is honest only because
both run the *same* SQLAlchemy Core statements - there is no second
implementation to drift. What SQLite cannot exercise is called out on the test
that would cover it.

The point of these is parity: a record written to a database must behave
exactly like one written to a file, chain and refusals included, or switching
backends silently changes what the product claims.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import db, record  # noqa: E402
from tests.test_record import NOW, match, results  # noqa: E402


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    """A throwaway database, wired up the way production is: by env var."""
    dsn = "sqlite:///" + str(tmp_path / "t.db").replace("\\", "/")
    monkeypatch.setenv("DATABASE_URL", dsn)
    db._engine = None                      # the module caches one engine
    eng = db.engine()
    db.create_all(eng)
    yield eng
    db._engine = None


# ------------------------------------------------------------------ wiring
def test_no_database_url_means_the_file_store(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db._engine = None
    assert record.backend(str(tmp_path)).startswith("file:")


def test_a_database_url_switches_the_backend(tmp_path, sqlite_db):
    assert record.backend(str(tmp_path)).startswith("db:")


def test_platform_connection_strings_are_accepted():
    """Heroku, Railway and Render still hand out the postgres:// form, and
    SQLAlchemy 2 rejects it outright."""
    assert db._normalise("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert db._normalise("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert db._normalise("sqlite:///x.db") == "sqlite:///x.db"


# ------------------------------------------------------------------ parity
def test_the_database_record_behaves_like_the_file_one(tmp_path, sqlite_db):
    rows = [match("Arsenal", "Chelsea", "2026-09-13T08:00:00+03:00"),
            match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
            match("Celta", "Malaga", "2026-09-13T19:00:00+03:00", div="SP1")]
    r = record.publish(rows, str(tmp_path), as_of=NOW)
    assert r["written"] == 2
    assert r["refused_already_started"] == 1
    assert r["backend"].startswith("db:")
    assert record.verify(str(tmp_path))["ok"]


def test_publishing_twice_adds_nothing(tmp_path, sqlite_db):
    rows = [match("Leeds", "Everton", "2026-09-13T18:00:00+03:00")]
    record.publish(rows, str(tmp_path), as_of=NOW)
    again = record.publish(rows, str(tmp_path), as_of=NOW)
    assert again["written"] == 0 and again["already_recorded"] == 1


def test_the_chain_continues_across_separate_publishes(tmp_path, sqlite_db):
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00")],
                   str(tmp_path), as_of=NOW)
    record.publish([match("Inter", "Roma", "2026-09-20T18:00:00+03:00", div="I1")],
                   str(tmp_path), as_of="2026-09-20T09:00:00+03:00")
    v = record.verify(str(tmp_path))
    assert v["ok"] and v["rows"] == 2


def test_summary_reads_the_database(tmp_path, sqlite_db):
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00",
                          p=(0.8, 0.15, 0.05))], str(tmp_path), as_of=NOW)
    s = record.summary(
        str(tmp_path), results([("E0", "2026-09-13", "Leeds", "Everton", 2, 0)]))
    assert s["published"] == 1 and s["settled"] == 1
    assert s["recent"][0]["correct"] is True


def test_a_row_edited_in_the_database_still_breaks_the_chain(tmp_path, sqlite_db):
    """The tamper-evidence is in the data, not in the file format."""
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
                    match("Celta", "Malaga", "2026-09-13T19:00:00+03:00",
                          div="SP1")], str(tmp_path), as_of=NOW)
    with sqlite_db.begin() as c:
        c.execute(db.predictions.update()
                  .where(db.predictions.c.home == "Leeds")
                  .values(pH=0.99))
    v = record.verify(str(tmp_path))
    assert not v["ok"] and v["broken_at"] == 0


def test_the_same_fixture_cannot_be_inserted_twice(tmp_path, sqlite_db):
    """The constraint, not the application, is what stops a double publish.

    Two workers reading an empty table at the same instant both decide to
    write; on Postgres the advisory lock serialises them, and this unique index
    is the backstop if anything ever bypasses it.
    """
    from sqlalchemy.exc import IntegrityError
    record.publish([match("Leeds", "Everton", "2026-09-13T18:00:00+03:00")],
                   str(tmp_path), as_of=NOW)
    row = dict(record.load(str(tmp_path)).iloc[0])
    row.pop("id", None)
    row["hash"] = "f" * 64                       # a different hash, same fixture
    with pytest.raises(IntegrityError):
        with sqlite_db.begin() as c:
            c.execute(db.predictions.insert(), [row])


# --------------------------------------------------------------- accounts
def test_the_account_tables_exist_and_hold_money_as_integers(sqlite_db):
    """Minor units, never floats: 0.1 + 0.2 is not 0.3 and a subscription
    priced in TZS has no business being a binary fraction."""
    from sqlalchemy import select
    with sqlite_db.begin() as c:
        uid = c.execute(db.users.insert().values(
            msisdn="+255700000001", display_name="Test",
            created_at=db.now())).inserted_primary_key[0]
        c.execute(db.subscriptions.insert().values(
            user_id=uid, plan="weekly", status="trial",
            started_at=db.now(), currency="TZS", amount_minor=200000,
            created_at=db.now()))
        got = c.execute(select(db.subscriptions.c.amount_minor)).scalar_one()
    assert got == 200000 and isinstance(got, int)


def test_a_repeated_payment_callback_cannot_bill_twice(sqlite_db):
    """Aggregators retry webhooks. The provider's reference is the idempotency
    key, enforced by the database rather than by remembering to check."""
    from sqlalchemy.exc import IntegrityError
    row = dict(provider="selcom", provider_ref="TX-1", status="paid",
               currency="TZS", amount_minor=200000, received_at=db.now())
    with sqlite_db.begin() as c:
        c.execute(db.payments.insert().values(**row))
    with pytest.raises(IntegrityError):
        with sqlite_db.begin() as c:
            c.execute(db.payments.insert().values(**row))


def test_health_reports_the_backend(sqlite_db, tmp_path, monkeypatch):
    h = db.healthy(sqlite_db)
    assert h["configured"] and h["ok"] and h["backend"] == "sqlite"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db._engine = None
    assert db.healthy()["backend"] == "files"


def test_an_unreachable_database_is_reported_not_raised():
    """A dead database must degrade to a health flag, not a stack trace on
    every request."""
    eng = db.engine("postgresql+psycopg://nobody@127.0.0.1:1/none")
    h = db.healthy(eng)
    assert h["configured"] and not h["ok"] and "error" in h
    db._engine = None
