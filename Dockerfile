# The prediction service: FastAPI, the engine, and the match data it fits on.
#
# Two kinds of match data, shipped two different ways:
#
#   * The repo bundle (data/) - the African, Nordic and Eastern European
#     divisions converted from other sources, the Tanzanian overlays, and the
#     committed record. Small and versioned with the code, so it is copied in.
#   * The football-data.co.uk pool - every European division and the fixtures
#     feed. It lives outside the repo and is refreshed in place, so it is
#     MOUNTED at /data/soccer rather than baked in.
#
# Get the second one wrong and nothing fails. The loader always merges the
# bundle with whatever FOOTBALL_DATA points at, so a missing mount quietly
# serves fifteen divisions instead of thirty - no Premier League, no La Liga -
# and every endpoint still answers 200. /api/health reports
# `data.missing_top10` precisely so that mistake is visible.
#
# The published record is the one piece of state that cannot be regenerated.
# Without a database it lives on the RECORD_ROOT volume; with DATABASE_URL it
# lives in Postgres - run `predict.py record-migrate` once when switching, or
# publishing restarts the hash chain and orphans everything already on file.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so an edit to the code does not re-resolve the world.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor/ ./predictor/
COPY service/ ./service/
COPY data/ ./data/
COPY predict.py ./

# FOOTBALL_DATA is the mounted European pool; the CLI reads the same variable,
# so `docker compose exec api python predict.py update` refreshes the mount.
ENV FOOTBALL_DATA=/data/soccer \
    RECORD_ROOT=/var/lib/predictor \
    PORT=8000

# Run as an unprivileged user. The only writable paths are the two volumes,
# created and owned here rather than by the container at start-up.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/soccer /var/lib/predictor \
    && chown -R app:app /data/soccer /var/lib/predictor
USER app

EXPOSE 8000

# Boot fits thirty divisions in about eleven seconds on a warm-up thread, so
# the start period allows for it. /api/health answers immediately and reports
# warm-up, data, database and live-score state rather than failing on them.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

# One worker by default. The engine holds the whole fitted pool in memory
# (~166 MB resident) and workers do not share it. Scale with replicas behind a
# proxy, and only with DATABASE_URL set: two workers appending to one record
# CSV corrupt its hash chain, and without Postgres's advisory lock two live-
# score refreshers can race for the same request of the daily budget.
CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
