#!/bin/sh
# Container entrypoint: get the data volume ready, then serve.
#
# Starts as root for exactly one reason - a freshly mounted Fly volume belongs
# to root and the service has to write to it - and drops to the unprivileged
# app user before running anything that touches the network.
set -eu

APP_UID=10001
DATA=/app/data
SEED=/app/seed/data
SOCCER="${FOOTBALL_DATA:-$DATA/soccer}"

mkdir -p "$DATA" "$SOCCER"

# First boot on an empty volume: copy in the league bundle, the Tanzanian
# overlays, the raw files the Tanzanian rebuild regenerates from, and the
# committed record. Only ever when empty. After that the volume is the source
# of truth, because it is where refreshes write and where the record grows.
if [ ! -d "$DATA/leagues" ]; then
  echo "first boot: seeding $DATA from the image"
  cp -a "$SEED/." "$DATA/"
fi

if [ -n "${RECORD_ROOT:-}" ]; then
  mkdir -p "$RECORD_ROOT/data/record"
  chown -R "$APP_UID:$APP_UID" "$RECORD_ROOT/data" 2>/dev/null || true
fi
chown -R "$APP_UID:$APP_UID" "$DATA" "$SOCCER" 2>/dev/null || true

as_app() {
  setpriv --reuid="$APP_UID" --regid="$APP_UID" --clear-groups \
    env HOME=/home/app "$@"
}

# The European pool (football-data.co.uk) lives outside git. On an empty
# volume it is downloaded - in the background, so the service answers health
# checks at once rather than failing a deploy while a hundred files download.
# Until it lands /api/health lists the missing leagues; the service notices the
# new files by itself and refits. The marker keeps the six-hourly refresh from
# writing the same files at the same time.
if [ -z "$(find "$SOCCER" -name '*.csv' -print -quit 2>/dev/null)" ]; then
  echo "European pool empty: downloading from football-data.co.uk in the background"
  touch "$SOCCER/.bootstrapping"
  (
    as_app python /app/predict.py update --since 2023 \
      || echo "European download failed - /api/health lists what is missing"
    rm -f "$SOCCER/.bootstrapping"
  ) &
fi

exec setpriv --reuid="$APP_UID" --regid="$APP_UID" --clear-groups \
  env HOME=/home/app \
  uvicorn service.app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
