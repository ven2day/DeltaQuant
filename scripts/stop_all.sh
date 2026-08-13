#!/usr/bin/env bash
# Stops whatever scripts/start_all.sh started.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

stop_pid_file() {
    local name="$1"
    local pid_file="$2"
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        kill "$(cat "$pid_file")"
        echo "Stopped $name (PID $(cat "$pid_file"))"
    else
        echo "$name not running"
    fi
    rm -f "$pid_file"
}

# Git Bash PIDs don't always map cleanly onto real Windows PIDs (same issue
# start_all.sh already documents for its own liveness check) -- a stale/
# coincidental PID in the .pid file can look dead via `kill -0` even though the
# real process is still bound to the port, so stop_pid_file() silently no-ops
# and the old process (with whatever stale settings/code it started with) just
# keeps running forever. Kill by the port it's actually listening on instead --
# that's the thing that's unambiguously true regardless of which shell's PID
# numbering is in play.
kill_by_port() {
    local name="$1"
    local port="$2"
    local pid
    pid="$(netstat -ano 2>/dev/null | grep ":$port " | grep LISTENING | awk '{print $NF}' | sort -u | head -1)"
    if [ -n "${pid:-}" ]; then
        if taskkill //F //PID "$pid" >/dev/null 2>&1; then
            echo "Stopped $name (PID $pid, port $port)"
        else
            echo "WARNING: found $name on port $port (PID $pid) but could not kill it" >&2
        fi
    else
        echo "$name not running (port $port closed)"
    fi
}

WEB_UI_PORT="$(grep -m1 '^WEB_UI_PORT=' env/.env.common 2>/dev/null | cut -d= -f2)"
WEB_UI_PORT="${WEB_UI_PORT:-8010}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

kill_by_port "backend" "$WEB_UI_PORT"
rm -f run/nse/backend.pid
stop_pid_file "forex worker" run/forex/backend.pid
kill_by_port "frontend" "$FRONTEND_PORT"
rm -f run/common/frontend.pid

if [ -f "$REPO_ROOT/compose.yml" ] && [ -f "$REPO_ROOT/env/.env.common" ]; then
    echo "Stopping docker compose services (deltaquant-postgres, market-history-db)..."
    # `down` never actually connects using this value, but compose still
    # requires POSTGRES_PASSWORD to be set to interpolate the file -- derive
    # it the same way start_all.sh does rather than hardcoding a placeholder.
    eval "$(uv run python -c "
import re, urllib.parse as u
text = open('env/.env.common').read()
m = re.search(r'^DATABASE_URL=(.+)\$', text, re.MULTILINE)
if m:
    parsed = u.urlparse(m.group(1).strip())
    print(f'export POSTGRES_PASSWORD={u.unquote(parsed.password or \"\")!r}')
")"
    docker compose -f "$REPO_ROOT/compose.yml" down
fi
