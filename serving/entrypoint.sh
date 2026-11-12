#!/bin/sh
# Serves the 'champion' model with `mlflow models serve` and automatically switches
# to a new version when the alias moves (new champion or rollback). Polls every 30 s.
MODEL_NAME="${MODEL_NAME:-AutoRouteModel}"
PORT="${PORT:-5001}"
HOST="${HOST:-0.0.0.0}"
CURRENT=""
PID=""
trap '[ -n "$PID" ] && kill $PID 2>/dev/null; exit 0' TERM INT
while true; do
  NEW=$(python /app/serving/get_champion.py 2>/dev/null)
  if [ -n "$NEW" ] && [ "$NEW" != "$CURRENT" ]; then
    echo "[entrypoint] champion is now version $NEW - (re)starting model server"
    if [ -n "$PID" ]; then kill $PID 2>/dev/null; wait $PID 2>/dev/null; fi
    mlflow models serve -m "models:/${MODEL_NAME}/${NEW}" -h "$HOST" -p "$PORT" --env-manager local &
    PID=$!
    CURRENT="$NEW"
  elif [ -z "$NEW" ] && [ -z "$CURRENT" ]; then
    echo "[entrypoint] no champion model yet - waiting (run the training pipeline)"
  fi
  sleep 30
done
