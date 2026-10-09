"""Say out loud when a source stops arriving.

A deployed service that quietly serves stale data is worse than one that is
down: nothing looks wrong. The refresh cycle checks itself afterwards, and
anything broken is posted to `ALERT_WEBHOOK_URL` - any endpoint that accepts
`{"text": "..."}`, which covers Slack, Discord (with /slack), Google Chat and
most self-hosted bridges.

Two rules keep it worth reading:

  * the same problem is not repeated more often than ALERT_REPEAT_HOURS, so a
    week-long outage is a handful of messages rather than a wall of them;
  * a problem that clears is cleared from the state, so when it comes back it
    is announced again rather than being swallowed as "already sent".

With no webhook configured nothing is sent and nothing fails: the message is
still printed to the log, which is where it is found locally.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone

REPEAT_HOURS = float(os.environ.get("ALERT_REPEAT_HOURS", "6") or 6)
TIMEOUT = 10


def webhook() -> str:
    return (os.environ.get("ALERT_WEBHOOK_URL") or "").strip()


def configured() -> bool:
    return bool(webhook())


def state_path(root: str) -> str:
    return os.path.join(root, "data", "record", "alerts.json")


def _load(root: str) -> dict:
    try:
        with open(state_path(root), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save(root: str, state: dict) -> None:
    os.makedirs(os.path.dirname(state_path(root)), exist_ok=True)
    with open(state_path(root), "w", encoding="utf-8") as fh:
        json.dump(state, fh)


def _utcnow(now=None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def post(url: str, text: str, transport=None) -> int:
    """POST {"text": ...}. Separate so tests never touch the network."""
    if transport is not None:
        return transport(url, text)
    from service import live
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT,
                                context=live._tls_context()) as r:
        return r.status


def notify(root: str, key: str, text: str, now=None, transport=None) -> dict:
    """Send one alert, unless the same key went out recently."""
    now = _utcnow(now)
    state = _load(root)
    prev = state.get(key)
    if prev:
        try:
            when = datetime.fromisoformat(prev["at"])
            if now - when < timedelta(hours=REPEAT_HOURS):
                return {"sent": False, "reason": "already alerted %s"
                        % prev["at"], "key": key}
        except (KeyError, TypeError, ValueError):
            pass
    print("ALERT [%s] %s" % (key, text), flush=True)
    url = webhook()
    if not url:
        state[key] = {"at": now.isoformat(), "sent": False, "text": text}
        _save(root, state)
        return {"sent": False, "reason": "no ALERT_WEBHOOK_URL set", "key": key}
    try:
        status = post(url, text, transport)
        ok = 200 <= int(status) < 300
        out = {"sent": ok, "status": int(status), "key": key}
        if not ok:
            out["reason"] = "webhook answered HTTP %s" % status
    except Exception as e:
        out = {"sent": False, "key": key,
               "reason": "webhook failed: %s: %s" % (type(e).__name__, e)}
    state[key] = {"at": now.isoformat(), "sent": bool(out["sent"]), "text": text}
    _save(root, state)
    return out


def clear(root: str, keys) -> int:
    """Forget problems that have cleared, so a recurrence is announced again."""
    state = _load(root)
    gone = [k for k in list(state) if k in set(keys)]
    for k in gone:
        state.pop(k, None)
    if gone:
        _save(root, state)
    return len(gone)


def check(root: str, rep: dict, failures=(), now=None, transport=None) -> dict:
    """Alert on a freshness report and on steps that failed this cycle.

    One message per problem source, plus one for a refresh step that raised -
    a step can fail while its file is still fresh enough to look fine.
    """
    sent, skipped = [], []
    for s in rep.get("sources", []):
        if s["level"] == "ok":
            continue
        text = ("Football Predictor: %s is %s (last %s). %s%s"
                % (s["label"], s["level"], s["last"],
                   (s["note"] + " ") if s.get("note") else "",
                   ("Fix: " + s["fix"]) if s.get("fix") else "")).strip()
        r = notify(root, "source:" + s["key"], text, now, transport)
        (sent if r["sent"] else skipped).append(r)
    for f in failures or ():
        r = notify(root, "step:" + str(f)[:40],
                   "Football Predictor refresh step failed: %s" % f,
                   now, transport)
        (sent if r["sent"] else skipped).append(r)
    healthy = ["source:" + s["key"] for s in rep.get("sources", [])
               if s["level"] == "ok"]
    cleared = clear(root, healthy)
    return {"sent": len(sent), "not_sent": len(skipped), "cleared": cleared,
            "configured": configured()}
