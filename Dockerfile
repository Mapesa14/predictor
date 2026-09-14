# The prediction service and the web app, served from one address.
#
# Two kinds of match data, shipped two different ways:
#
#   * The repo bundle (data/) - the African, Nordic and Eastern European
#     divisions, the Tanzanian overlays, the raw files the Tanzanian rebuild
#     regenerates from, and the committed record. Copied into the image as a
#     seed; deploy/start.sh copies it onto the data volume on first boot.
#   * The football-data.co.uk pool - every European division and the fixtures
#     feed. It lives outside git, so it is downloaded onto the volume on first
#     boot and refreshed in place from then on.
#
# Get the second one wrong and nothing fails: the loader merges the bundle with
# an empty pool, serves fifteen divisions instead of thirty, and every endpoint
# answers 200. /api/health reports `data.missing_top10` so it is visible.
#
# The record is the one piece of state that cannot be regenerated. It lives on
# the volume; with DATABASE_URL it moves into Postgres - run
# `predict.py record-migrate` once when switching.

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
COPY predict.py ./
COPY web/dist/ ./web/dist/
COPY data/ ./seed/data/
COPY deploy/start.sh ./deploy/start.sh

# The sed is belt and braces for a Windows checkout: .gitattributes keeps the
# script LF, but a CRLF entrypoint fails in a way that looks like a missing
# file rather than a line-ending problem.
RUN sed -i 's/\r$//' /app/deploy/start.sh \
    && chmod +x /app/deploy/start.sh \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /data/soccer /var/lib/predictor \
    && chown -R app:app /app/data /data/soccer /var/lib/predictor

# Package-relative paths (data/leagues, data/manual) resolve to /app/data, the
# volume, so refreshes write where they persist. docker-compose overrides these.
ENV FOOTBALL_DATA=/app/data/soccer \
    RECORD_ROOT=/app \
    PORT=8000

EXPOSE 8000

# /api/health answers as soon as the process is up and reports warm-up, data,
# database and live-score state rather than failing on them.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

# No USER line on purpose: the entrypoint starts as root only to take
# ownership of a freshly mounted volume, then drops to uid 10001 before it runs
# the download or the server. One worker - the fitted models live in memory
# and workers do not share them; scale with machines, and only with Postgres.
CMD ["/app/deploy/start.sh"]
