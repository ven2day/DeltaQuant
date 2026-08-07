#!/usr/bin/env bash
# Starts the ₹DeltaQuant backend (trading loop + FastAPI/WebSocket) and the
# Next.js frontend together, on a VPS/Linux host. Run from the repo root:
#   ./scripts/start_all.sh
# Stop everything with scripts/stop_all.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs run

if [ ! -f .env ]; then
    echo "ERROR: .env not found in $REPO_ROOT — copy .env.example to .env and fill it in first." >&2
    exit 1
fi

is_running() {
    local pid_file="$1"
    [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

# --- Backend: trading loop + FastAPI/WebSocket (requires ENABLE_WEB_UI=true, WEB_UI_HOST=0.0.0.0 in .env) ---
if is_running run/backend.pid; then
    echo "Backend already running (PID $(cat run/backend.pid))"
else
    echo "Starting backend..."
    nohup uv run --extra web python scripts/run_live_trading.py > logs/backend.log 2>&1 &
    echo $! > run/backend.pid
    echo "Backend started (PID $!) — logs/backend.log"
fi

# --- Frontend: Next.js dashboard ---
if is_running run/frontend.pid; then
    echo "Frontend already running (PID $(cat run/frontend.pid))"
else
    echo "Starting frontend..."
    (
        cd web
        # Use `npm run build && npm run start` instead of `dev` for a production
        # run (faster, no hot-reload overhead) once you're past active development.
        nohup npm run dev > ../logs/frontend.log 2>&1 &
        echo $! > ../run/frontend.pid
    )
    echo "Frontend started (PID $(cat run/frontend.pid)) — logs/frontend.log"
fi

sleep 2
echo ""
echo "Backend log:  tail -f $REPO_ROOT/logs/backend.log"
echo "Frontend log: tail -f $REPO_ROOT/logs/frontend.log"
echo "Stop both:    ./scripts/stop_all.sh"
