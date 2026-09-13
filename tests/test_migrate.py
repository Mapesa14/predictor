"""Moving the record from a file into Postgres must not restart its chain.

Switching on DATABASE_URL without this would begin an empty table, orphaning
every prediction already published - the record's whole history, which is the
only thing that makes it evidence. These pin the migration: rows cross
verbatim, the chain verifies on the far side, re-running is harmless, and the
two cases that cannot be reconciled automatically are refused.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import db, record  # noqa: E402
from tests.test_record import NOW, match  # noqa: E402

TWO = [match("Leeds", "Everton", "2026-09-13T18:00:00+03:00"),
       match("Celta", "Malaga", "2026-09-13T19:00:00+03:00", div="SP1")]


@pytest.fixture(autouse=True)
def fresh_engine():
    db._engine = None
    yield
    db._engine = None


def use_files(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db._engine = None


def use_db(tmp_path, monkeypatch):
    dsn = "sqlite:///" + str(tmp_path / "m.db").replace("\\", "/")
    monkeypatch.setenv("DATABASE_URL", dsn)
    db._engine = None
    return db.engine()


def test_the_chain_crosses_verbatim(tmp_path, monkeypatch):
    root = str(tmp_path)
    use_files(monkeypatch)
    record.publish(TWO, root, as_of=NOW)
    file_hashes = record.FileStore(root).read()["hash"].tolist()

    eng = use_db(tmp_path, monkeypatch)
    r = record.migrate(root)
    assert r["ok"] and r["migrated"] == 2
    assert record.DbStore(eng).read()["hash"].tolist() == file_hashes
    v = record.verify(root)                 # now reading the database
    assert v["ok"] and v["rows"] == 2


def test_publishing_continues_the_migrated_chain(tmp_path, monkeypatch):
    """The first new prediction after the switch must link to the last
    migrated one, not start again from GENESIS."""
    root = str(tmp_path)
    use_files(monkeypatch)
    record.publish(TWO[:1], root, as_of=NOW)
    use_db(tmp_path, monkeypatch)
    record.migrate(root)
    record.publish(TWO[1:], root, as_of=NOW)
    v = record.verify(root)
    assert v["ok"] and v["rows"] == 2


def test_running_it_twice_changes_nothing(tmp_path, monkeypatch):
    root = str(tmp_path)
    use_files(monkeypatch)
    record.publish(TWO, root, as_of=NOW)
    use_db(tmp_path, monkeypatch)
    record.migrate(root)
    again = record.migrate(root)
    assert again["ok"] and again["migrated"] == 0
    assert "already migrated" in again["reason"]


def test_rows_published_to_the_file_since_are_appended(tmp_path, monkeypatch):
    root = str(tmp_path)
    use_files(monkeypatch)
    record.publish(TWO[:1], root, as_of=NOW)
    use_db(tmp_path, monkeypatch)
    record.migrate(root)

    use_files(monkeypatch)
    record.publish(TWO[1:], root, as_of=NOW)
    use_db(tmp_path, monkeypatch)
    r = record.migrate(root)
    assert r["ok"] and r["migrated"] == 1 and r["rows"] == 2


def test_a_diverged_database_is_refused(tmp_path, monkeypatch):
    """Two chains that do not share a history cannot be spliced honestly."""
    root = str(tmp_path)
    use_files(monkeypatch)
    record.publish(TWO[:1], root, as_of=NOW)
    use_db(tmp_path, monkeypatch)
    record.migrate(root)
    record.publish(TWO[1:], root, as_of=NOW)       # written to the database only

    r = record.migrate(root)
    assert r["ok"] is False and r["migrated"] == 0
    assert "diverged" in r["reason"]
    assert record.verify(root)["rows"] == 2          # nothing removed or added


def test_a_broken_file_record_is_not_copied(tmp_path, monkeypatch):
    root = str(tmp_path)
    use_files(monkeypatch)
    record.publish(TWO, root, as_of=NOW)
    p = record.path(root)
    text = open(p, encoding="utf-8").read().replace("0.5", "0.9", 1)
    open(p, "w", encoding="utf-8", newline="").write(text)

    eng = use_db(tmp_path, monkeypatch)
    r = record.migrate(root)
    assert r["ok"] is False and r["migrated"] == 0 and "broken" in r["reason"]
    db.create_all(eng)
    assert record.DbStore(eng).read().empty


def test_there_must_be_a_database_to_migrate_into(tmp_path, monkeypatch):
    use_files(monkeypatch)
    with pytest.raises(RuntimeError):
        record.migrate(str(tmp_path))
