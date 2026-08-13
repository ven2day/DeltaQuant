#!/usr/bin/env bash
# Start the DeltaQuant postgres/timescale containers, the canonical NSE worker,
# and the shared Next.js frontend from the repo root -- one command, in order.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/nse logs/common logs/forex run/nse run/common run/forex

for profile in env/.env.common env/.env.nse env/.env.forex.practice; do
    if [ ! -f "$profile" ]; then
        echo "ERROR: $profile is missing; copy its .example template and configure it." >&2
        exit 1
    fi
done

is_running() {
    local pid_file="$1"
    [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

# PID-file liveness (is_running) is unreliable here: Git Bash PIDs don't
# always map cleanly onto real Windows PIDs, so a stale/coincidental PID can
# look "alive" via kill -0 even though the actual backend/frontend already
# exited (or vice versa) -- this bit us during testing (a stale frontend.pid
# caused a duplicate frontend to start on port 3001). Port liveness is the
# thing that actually matters, so gate starts on that instead; the PID file
# is kept only so stop_all.sh has something to kill.
port_open() {
    local port="$1"
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && exec 3<&- 3>&-
}

WEB_UI_PORT="$(grep -m1 '^WEB_UI_PORT=' env/.env.common | cut -d= -f2)"
WEB_UI_PORT="${WEB_UI_PORT:-8010}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# --- 1. Docker: postgres + timescaledb -------------------------------------
# POSTGRES_PASSWORD/USER/DB are derived from env/.env.common's existing
# DATABASE_URL (never duplicated as a literal secret into compose.yml, which
# is tracked in git) so the app's connection string keeps working unchanged.
echo "Deriving Postgres credentials from env/.env.common's DATABASE_URL..."
eval "$(uv run python -c "
import re, urllib.parse as u
text = open('env/.env.common').read()
m = re.search(r'^DATABASE_URL=(.+)\$', text, re.MULTILINE)
if not m:
    raise SystemExit('DATABASE_URL not found in env/.env.common')
parsed = u.urlparse(m.group(1).strip())
print(f'export POSTGRES_USER={u.unquote(parsed.username or \"postgres\")!r}')
print(f'export POSTGRES_PASSWORD={u.unquote(parsed.password or \"\")!r}')
print(f'export POSTGRES_DB={(parsed.path.lstrip(\"/\").split(\"?\")[0] or \"postgres\")!r}')
")"

echo "Starting docker compose services (deltaquant-postgres, market-history-db)..."
docker compose -f compose.yml up -d

wait_for_healthy() {
    local container="$1"
    local waited=0
    local timeout=60
    while [ "$waited" -lt "$timeout" ]; do
        status="$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo "missing")"
        if [ "$status" = "healthy" ]; then
            echo "$container is healthy."
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "ERROR: $container did not become healthy within ${timeout}s (last status: $status)" >&2
    return 1
}

wait_for_healthy deltaquant-postgres
wait_for_healthy deltaquant-market-history

# --- 2. Backend --------------------------------------------------------------
if port_open "$WEB_UI_PORT"; then
    echo "NSE backend already running (port $WEB_UI_PORT open)"
else
    echo "Starting NSE backend..."
    nohup uv run --extra web deltaquant-nse > logs/nse/backend.log 2>&1 &
    echo $! > run/nse/backend.pid
    echo "NSE backend started (PID $!) - logs/nse/backend.log"
fi

# --- 2b. Forex worker (headless) --------------------------------------------
# The Forex worker has no HTTP port of its own: it runs the OANDA paper cycle and
# publishes read-only snapshots to run/FOREX, which the NSE backend's dashboard
# server aggregates into the Forex tab. MARKET=FOREX makes settings load the
# env/.env.forex.practice profile (OANDA Practice, execution disabled). Liveness
# is PID-only here since there is no port to probe.
if [ -f run/forex/backend.pid ] && kill -0 "$(cat run/forex/backend.pid)" 2>/dev/null; then
    echo "Forex worker already running (PID $(cat run/forex/backend.pid))"
else
    echo "Starting Forex worker..."
    # --no-sync: the NSE backend (started above) holds deltaquant-nse.exe, so a
    # normal `uv run` re-link step fails to overwrite it on Windows. Dependencies
    # are already synced by the NSE start, so skip the sync here.
    MARKET=FOREX nohup uv run --no-sync deltaquant-forex > logs/forex/backend.log 2>&1 &
    echo $! > run/forex/backend.pid
    echo "Forex worker started (PID $!) - logs/forex/backend.log"
fi

# --- 3. Frontend ---------------------------------------------------------------
if port_open "$FRONTEND_PORT"; then
    echo "Frontend already running (port $FRONTEND_PORT open)"
else
    echo "Starting frontend..."
    (
        cd web
        nohup npm run dev > ../logs/common/frontend.log 2>&1 &
        echo $! > ../run/common/frontend.pid
    )
    echo "Frontend started (PID $(cat run/common/frontend.pid)) - logs/common/frontend.log"
fi

sleep 2
echo
echo "===================================================================="
echo " DeltaQuant is up:"
echo "   deltaquant-postgres        healthy (127.0.0.1:5432)"
echo "   deltaquant-market-history  healthy (127.0.0.1:5433)"
echo "   NSE backend                port $WEB_UI_PORT open  logs/nse/backend.log"
echo "   Forex worker               headless (run/FOREX)   logs/forex/backend.log"
echo "   Frontend                   port $FRONTEND_PORT open  logs/common/frontend.log"
echo "===================================================================="
echo "Stop everything: ./scripts/stop_all.sh"
