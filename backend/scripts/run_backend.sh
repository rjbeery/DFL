#!/usr/bin/env bash
# backend/scripts/run_backend.sh
# ─────────────────────────────────────────────────────────────────────────────
# Start the DFL FastAPI backend on a remote VM.
#
# Usage:
#   chmod +x run_backend.sh
#   ./run_backend.sh            # loads .env from ../(.env) by default
#   DFL_ENV=/path/.env ./run_backend.sh
#
# The script activates the backend venv, loads the .env file, then launches
# uvicorn. Suitable for manual startup and as the ExecStart in the systemd unit.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

# ── Load .env ──────────────────────────────────────────────────────────────
ENV_FILE="${DFL_ENV:-$BACKEND_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
    # Export each non-comment, non-blank line.
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[dfl-backend] Loaded env from $ENV_FILE"
else
    echo "[dfl-backend] WARNING: $ENV_FILE not found — using defaults / existing env."
fi

# ── Activate venv ──────────────────────────────────────────────────────────
VENV="$BACKEND_DIR/.venv"
if [[ ! -f "$VENV/bin/python" ]]; then
    echo "[dfl-backend] ERROR: venv not found at $VENV"
    echo "  Run:  python3.10 -m venv $VENV && $VENV/bin/pip install -r $BACKEND_DIR/requirements.txt"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── Resolve bind address ───────────────────────────────────────────────────
HOST="${DFL_BIND_HOST:-127.0.0.1}"
PORT="${DFL_BIND_PORT:-8000}"

echo "[dfl-backend] Starting on $HOST:$PORT  (workdir: $BACKEND_DIR)"

# ── Launch uvicorn ─────────────────────────────────────────────────────────
cd "$BACKEND_DIR"
exec uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --no-access-log
