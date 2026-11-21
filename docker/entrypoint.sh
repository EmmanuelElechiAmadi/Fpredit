#!/bin/sh
# Container entrypoint.
#
# Normal (web) boot: serve the FastAPI app.
# Cron boot: when CRON_REFRESH=1 is set (Render/Railway scheduled job), run the
# weekly data refresh instead and exit when it is done.
set -e

if [ "${CRON_REFRESH:-0}" = "1" ]; then
  echo "[entrypoint] running weekly data refresh (CRON_REFRESH=1)"
  exec python3 scripts/refresh_data.py --seasons 1 --strict "$@"
fi

echo "[entrypoint] starting web server on :${PORT:-8000}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
