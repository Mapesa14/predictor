"""Read the repo's .env into the environment, for running without Docker.

docker compose reads .env by itself; a plain `uvicorn` or `python predict.py`
does not. So a key pasted into .env worked in one and was silently missing in
the other - live scores simply stayed off, with nothing to say why. This fills
that gap and deliberately does nothing more:

  * a variable already set in the real environment always wins, so a hosting
    platform's secret store is never overridden by a stray file;
  * an empty value is treated as unset - the template's `LIVE_API_KEY=` slot
    must not become an empty string, and an empty FOOTBALL_DATA must not
    replace the default data path;
  * values are never printed, logged or returned, only the names loaded;
  * only the repo-root .env is read, which .gitignore keeps out of git and
    .dockerignore keeps out of images;
  * it does nothing under pytest unless forced, so a real DATABASE_URL in a
    developer's .env can never make the test suite write to that database.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path() -> str:
    return os.path.join(REPO, ".env")


def load(file: str | None = None, force: bool = False) -> list:
    """Set unset variables from the file. Returns the names it set."""
    if "pytest" in sys.modules and not force:
        return []
    file = file or path()
    if not os.path.isfile(file):
        return []
    loaded = []
    with open(file, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if not name.isidentifier() or name in os.environ or value == "":
                continue
            os.environ[name] = value
            loaded.append(name)
    return loaded
