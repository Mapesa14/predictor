"""The public track record: what was predicted, before kick-off, unedited.

A prediction that can be revised after the result is worthless as evidence, and
a record assembled after the fact proves nothing at all. So this module does
exactly two things and refuses to do anything else:

  * `publish` appends a row for every fixture whose kick-off is still ahead. A
    fixture that has already started is refused, not recorded.
  * nothing ever rewrites a row. Settlement is a *join* to the results the rest
    of the engine already loads, recomputed on every read - there is no stored
    outcome for anyone to quietly correct.

Every row carries a hash of itself chained to the hash of the row before it, so
"unedited" is a claim that can be checked rather than one taken on trust:
change any past row and `verify` names the first row where the chain breaks.
Appending stays cheap, which is the property a per-file checksum would lose.

The store is a plain append-only CSV at `data/record/predictions.csv`. Accuracy
comes from `backtest.score` and `backtest.calibration` - the same functions the
walk-forward evaluation uses, so the published record is measured exactly the
way the README's headline numbers were.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime

import numpy as np
import pandas as pd

from . import backtest, db, leagues, loader

# Written once per fixture and never touched again. The market columns hold the
# de-vigged closing probabilities *as they stood at publication*: the price
# moves, and comparing against one looked up afterwards would flatter whichever
# side was read last.
COLUMNS = [
    "published_at", "div", "league", "date", "kickoff", "home", "away",
    "pH", "pD", "pA", "pOver25", "pBTTS", "score", "xgH", "xgA",
    "mk1", "mkX", "mk2", "prev", "hash",
]

GENESIS = "0" * 64


def path(root: str) -> str:
    return os.path.join(root, "data", "record", "predictions.csv")


# ------------------------------------------------------------- hash chain
def _canon(row: dict) -> str:
    """One row as a deterministic string: fixed order, fixed-width floats."""
    parts = []
    for c in COLUMNS:
        if c in ("prev", "hash"):
            continue
        v = row.get(c, "")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = ""
        elif isinstance(v, float):
            v = "%.6f" % v
        parts.append("%s=%s" % (c, v))
    return "|".join(parts)


def _link(prev: str, row: dict) -> str:
    return hashlib.sha256((prev + "|" + _canon(row)).encode("utf-8")).hexdigest()


def _check_chain(df: pd.DataFrame) -> dict:
    """Walk a chain held in any frame, naming the first row that breaks it."""
    if df.empty:
        return {"ok": True, "rows": 0, "note": "nothing published yet"}
    prev = GENESIS
    for i, row in df.reset_index(drop=True).iterrows():
        r = {c: row[c] for c in COLUMNS if c not in ("prev", "hash")}
        if str(row["prev"]) != prev or _link(prev, r) != str(row["hash"]):
            return {"ok": False, "rows": int(len(df)), "broken_at": int(i),
                    "note": "row %d (%s v %s) has been altered, reordered or "
                            "removed since it was published"
                            % (i, row["home"], row["away"])}
        prev = str(row["hash"])
    return {"ok": True, "rows": int(len(df)),
            "note": "every row matches the hash chain"}


def verify(root: str) -> dict:
    """Walk the chain in whichever store is configured."""
    return _check_chain(load(root))


def migrate(root: str, eng=None) -> dict:
    """Copy a file record into the database without restarting its chain.

    Switching on DATABASE_URL otherwise begins an empty table: publishing
    carries on from GENESIS, and every prediction already on file is orphaned
    from the record the product shows - its entire history, which is the only
    thing that makes it worth anything. So the rows move across verbatim, with
    their original `published_at`, `prev` and `hash`, and the chain is checked
    on both sides.

    Refuses rather than guesses in the two cases that cannot be reconciled
    automatically: a file whose own chain is already broken, and a database
    that holds rows the file does not. Safe to run repeatedly; rows published
    to the file since the last run are appended.
    """
    eng = eng or db.engine()
    if eng is None:
        raise RuntimeError("no DATABASE_URL set - there is no database to "
                           "migrate into")
    src = FileStore(root).read()
    src_chain = _check_chain(src)
    if not src_chain["ok"]:
        return {"ok": False, "migrated": 0, "rows": 0,
                "reason": "the file record is broken, so nothing was copied: "
                          + src_chain["note"]}
    db.create_all(eng)
    dst_store = DbStore(eng)
    dst = dst_store.read()
    src_h = [str(h) for h in src["hash"]] if len(src) else []
    dst_h = [str(h) for h in dst["hash"]] if len(dst) else []
    if dst_h == src_h:
        return {"ok": True, "migrated": 0, "rows": len(dst_h),
                "reason": "already migrated - the database holds exactly the "
                          "file's %d rows" % len(dst_h)}
    if len(dst_h) > len(src_h) or src_h[:len(dst_h)] != dst_h:
        return {"ok": False, "migrated": 0, "rows": len(dst_h),
                "reason": "the database record has diverged from the file "
                          "(%d rows there, %d on file, not a common prefix); "
                          "refusing to splice two chains together"
                          % (len(dst_h), len(src_h))}
    rows = []
    for _, r in src.iloc[len(dst_h):].iterrows():
        row = {}
        for c in COLUMNS:
            v = r[c]
            row[c] = None if isinstance(v, float) and pd.isna(v) else v
        rows.append(row)
    dst_store.append(rows)
    after = _check_chain(dst_store.read())
    return {"ok": after["ok"], "migrated": len(rows), "rows": after["rows"],
            "reason": after["note"]}


# ---------------------------------------------------------------- storage
_NUM = ["pH", "pD", "pA", "pOver25", "pBTTS", "xgH", "xgA", "mk1", "mkX", "mk2"]


class FileStore:
    """An append-only CSV. Right for one machine, wrong for two.

    Two workers appending to the same file interleave their writes and break
    the chain, which is the whole reason the database backend exists.
    """

    def __init__(self, root: str):
        self.root = root
        self.path = path(root)

    def read(self) -> pd.DataFrame:
        if not os.path.isfile(self.path):
            return pd.DataFrame(columns=COLUMNS)
        df = pd.read_csv(self.path, dtype={"prev": str, "hash": str,
                                           "score": str, "date": str})
        for c in _NUM:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def append(self, rows: list) -> int:
        if not rows:
            return 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(
            self.path, mode="a", header=not os.path.isfile(self.path),
            index=False)
        return len(rows)

    def describe(self) -> str:
        return "file:%s" % self.path


class DbStore:
    """The same rows in a table, for when more than one process writes them.

    Reading the last hash and inserting after it has to be one atomic step or
    two publishers produce two rows claiming the same predecessor. Postgres
    gets a transaction-scoped advisory lock; SQLite serialises writers itself.
    The unique index on `hash` is the backstop that turns a lost race into a
    failed insert rather than a silently forked chain.
    """

    LOCK_ID = 4711_2026        # arbitrary, but stable: the record's own lock

    def __init__(self, eng):
        self.engine = eng

    def read(self) -> pd.DataFrame:
        from sqlalchemy import select
        cols = [db.predictions.c[c] for c in COLUMNS]
        with self.engine.connect() as c:
            rows = c.execute(
                select(*cols).order_by(db.predictions.c.id)).mappings().all()
        df = pd.DataFrame([dict(r) for r in rows], columns=COLUMNS)
        for col in _NUM:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("prev", "hash", "score", "date"):
            df[col] = df[col].astype(object).where(df[col].notna(), "")
        return df

    def append(self, rows: list) -> int:
        if not rows:
            return 0
        with self.engine.begin() as c:
            c.execute(db.predictions.insert(), rows)
        return len(rows)

    def lock(self, conn) -> None:
        if self.engine.dialect.name == "postgresql":
            from sqlalchemy import text
            conn.execute(text("SELECT pg_advisory_xact_lock(:k)"),
                         {"k": self.LOCK_ID})

    def describe(self) -> str:
        return "db:%s" % self.engine.dialect.name


def store(root: str):
    """The file store, unless a database is configured.

    `root` is still the argument every caller passes, so switching backends is
    one environment variable and no code change anywhere upstream.
    """
    eng = db.engine()
    return DbStore(eng) if eng is not None else FileStore(root)


def load(root: str) -> pd.DataFrame:
    return store(root).read()


def backend(root: str) -> str:
    return store(root).describe()


def _key(div, date, home, away) -> tuple:
    return (str(div), str(date), str(home), str(away))


def publish(rows, root: str, as_of: datetime | None = None) -> dict:
    """Append predictions for fixtures that have not kicked off yet.

    `rows` are the slate's own match dictionaries. Returns what was written
    and, just as importantly, what was refused and why: a silent skip here
    would hide the one thing the record exists to prove.
    """
    now = pd.Timestamp(as_of or datetime.now())
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    st = store(root)
    have = st.read()
    seen = {_key(r["div"], r["date"], r["home"], r["away"])
            for _, r in have.iterrows()} if len(have) else set()
    prev = str(have["hash"].iloc[-1]) if len(have) else GENESIS

    written, started, dup, undated = [], 0, 0, 0
    for m in rows:
        ko = pd.to_datetime(m.get("date"), errors="coerce", utc=True)
        if pd.isna(ko):
            # No published kick-off means no way to prove the prediction came
            # first. Those fixtures stay out of the record entirely.
            undated += 1
            continue
        if ko <= now:
            started += 1
            continue
        day = ko.tz_convert(None).strftime("%Y-%m-%d")
        k = _key(m["div"], day, m["home"], m["away"])
        if k in seen:
            dup += 1
            continue
        mk = m.get("market") or {}
        row = {
            "published_at": now.isoformat(),
            "div": m["div"],
            "league": m.get("league") or leagues.name(m["div"]),
            "date": day,
            "kickoff": ko.isoformat(),
            "home": m["home"], "away": m["away"],
            "pH": float(m["p"]["1"]), "pD": float(m["p"]["X"]),
            "pA": float(m["p"]["2"]),
            "pOver25": _f(m.get("o25")), "pBTTS": _f(m.get("btts")),
            "score": m.get("score", ""),
            "xgH": _side(m.get("xg"), 0), "xgA": _side(m.get("xg"), 1),
            "mk1": _f(mk.get("1")), "mkX": _f(mk.get("X")),
            "mk2": _f(mk.get("2")),
        }
        row["prev"] = prev
        row["hash"] = _link(prev, row)
        prev = row["hash"]
        seen.add(k)
        written.append(row)

    st.append(written)
    return {"written": len(written), "already_recorded": dup,
            "refused_already_started": started, "refused_no_kickoff": undated,
            "file": path(root), "backend": st.describe(),
            "total": int(len(have)) + len(written)}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _side(xg, i):
    """'1.73-1.05' -> one side of it."""
    try:
        return float(str(xg).split("-")[i])
    except (AttributeError, IndexError, ValueError, TypeError):
        return np.nan


# -------------------------------------------------------------- settlement
def settle(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join published predictions to what actually happened.

    Recomputed on every read rather than stored, so there is no settled-outcome
    column anyone could edit, and a result that is later corrected at source
    flows straight through.
    """
    out = preds.copy()
    if out.empty:
        for c in ("FTHG", "FTAG", "FTR", "over25", "btts"):
            out[c] = pd.Series(dtype="float64" if c != "FTR" else "object")
        return out
    r = results[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    r["day"] = pd.to_datetime(r["Date"]).dt.strftime("%Y-%m-%d")
    r = r.drop_duplicates(subset=["Div", "day", "HomeTeam", "AwayTeam"])
    m = out.merge(r, left_on=["div", "date", "home", "away"],
                  right_on=["Div", "day", "HomeTeam", "AwayTeam"], how="left")
    played = m["FTHG"].notna() & m["FTAG"].notna()
    m["FTR"] = np.where(m["FTHG"] > m["FTAG"], "H",
                        np.where(m["FTHG"] < m["FTAG"], "A", "D"))
    m.loc[~played, "FTR"] = None
    m["over25"] = np.where((m["FTHG"] + m["FTAG"]) > 2.5, 1.0, 0.0)
    m["btts"] = np.where((m["FTHG"] > 0) & (m["FTAG"] > 0), 1.0, 0.0)
    m.loc[~played, ["over25", "btts"]] = np.nan
    actual = (m["FTHG"].where(played, -1).astype(int).astype(str) + "-" +
              m["FTAG"].where(played, -1).astype(int).astype(str))
    m["exact"] = np.where(played, m["score"].astype(str) == actual, np.nan)
    return m


# Below this many priced rows the model-versus-price comparison is noise, and
# quoting it would invite exactly the over-reading the comparison exists to
# prevent. Say nothing instead.
MIN_PRICED = 20


def _market_scores(done: pd.DataFrame) -> dict:
    """The same accuracy measures for the price, on the rows that carry one."""
    d = done.dropna(subset=["mk1", "mkX", "mk2"])
    if len(d) < MIN_PRICED:
        return {}
    p = d[["mk1", "mkX", "mk2"]].to_numpy()
    idx = d["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    hit = np.clip(p[np.arange(len(p)), idx], 1e-9, 1)
    return {
        "n_with_price": int(len(d)),
        "logloss_market": float(-np.mean(np.log(hit))),
        "rps_market": float(np.mean([backtest.rps(p[i], idx[i])
                                     for i in range(len(p))])),
        "acc_market": float(np.mean(p.argmax(1) == idx)),
        # the model's own numbers on exactly the same subset, so the two are
        # comparable - scoring the model on everything and the price on the
        # rows it happens to cover is the oldest trick in the genre
        "logloss_model_same_rows": float(-np.mean(np.log(np.clip(
            d[["pH", "pD", "pA"]].to_numpy()[np.arange(len(d)), idx], 1e-9, 1)))),
    }


def summary(root: str, results: pd.DataFrame, bins: int = 8) -> dict:
    """Headline accuracy, per competition and calibration - or an honest empty.

    `pending` is reported alongside `settled` on purpose. A record that quietly
    drops its unsettled rows can be made to say whatever its publisher wants.
    """
    preds = load(root)
    chain = verify(root)
    empty = {"published": 0, "settled": 0, "pending": 0, "chain": chain,
             "overall": {}, "market": {}, "by_league": [], "calibration": [],
             "recent": [], "first_published": None}
    if preds.empty:
        return empty

    df = settle(preds, results)
    done = df[df["FTR"].notna()].copy()
    out = dict(empty)
    out.update({
        "published": int(len(df)),
        "settled": int(len(done)),
        "pending": int(len(df) - len(done)),
        "first_published": str(df["published_at"].min()),
        "chain": chain,
    })
    if not len(done):
        return out

    out["overall"] = backtest.score(done)
    out["market"] = _market_scores(done)

    for d, g in done.groupby("div"):
        s = backtest.score(g)
        if not s:
            continue
        out["by_league"].append({
            "div": d, "league": leagues.name(d), "country": leagues.country(d),
            "n": int(len(g)), "logloss": s.get("logloss_1x2"),
            "rps": s.get("rps"), "acc": s.get("acc"),
        })
    out["by_league"].sort(key=lambda x: -x["n"])

    # Calibration on the favourite's probability: the number a reader can check
    # against their own memory of the results.
    p = done[["pH", "pD", "pA"]].to_numpy()
    done["pTop"] = p.max(axis=1)
    done["hitTop"] = (pd.Series(p.argmax(axis=1), index=done.index)
                      .map({0: "H", 1: "D", 2: "A"}) == done["FTR"]).astype(float)
    cal = backtest.calibration(done, "pTop", "hitTop", bins=bins)
    out["calibration"] = [
        {"bin": str(r["bin"]), "n": int(r["n"]),
         "predicted": float(r["predicted"]), "realised": float(r["realised"])}
        for _, r in cal.iterrows() if r["n"]]

    recent = done.sort_values("kickoff", ascending=False).head(40)
    out["recent"] = [{
        "div": r["div"], "league": r["league"], "kickoff": r["kickoff"],
        "published_at": r["published_at"],
        "home": r["home"], "away": r["away"],
        "p": {"1": _f(r["pH"]), "X": _f(r["pD"]), "2": _f(r["pA"])},
        "pick": {0: "1", 1: "X", 2: "2"}[int(np.argmax([r["pH"], r["pD"], r["pA"]]))],
        "result": r["FTR"],
        "score": "%d-%d" % (r["FTHG"], r["FTAG"]),
        "predicted_score": r["score"],
        "correct": bool(r["hitTop"]),
        "exact": bool(r["exact"]) if not pd.isna(r["exact"]) else False,
    } for _, r in recent.iterrows()]
    return out


def results_pool(data_root: str) -> pd.DataFrame:
    """The same results the engine fits on, for settling against."""
    return loader.load(data_root)
