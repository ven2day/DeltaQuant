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

stop_pid_file "backend" run/backend.pid
stop_pid_file "frontend" run/frontend.pid
