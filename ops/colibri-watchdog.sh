#!/usr/bin/env bash
set -eu

service=${COLI_WATCHDOG_SERVICE:-colibri-server.service}
url=${COLI_WATCHDOG_URL:-http://127.0.0.1:8000}
model=${COLI_WATCHDOG_MODEL:-deepseek-chat}
warmup=${COLI_WATCHDOG_WARMUP_SECONDS:-180}
probe_timeout=${COLI_WATCHDOG_PROBE_TIMEOUT:-20}
state_dir=${COLI_WATCHDOG_STATE_DIR:-"$HOME/.colibri"}
dry_run=${COLI_WATCHDOG_DRY_RUN:-0}
diagnostics=${COLI_WATCHDOG_DIAGNOSTICS:-1}

mkdir -p "$state_dir"
exec 9>"$state_dir/watchdog.lock"
flock -n 9 || exit 0

systemctl --user is-active --quiet "$service" || exit 0

active_us=$(systemctl --user show "$service" -p ActiveEnterTimestampMonotonic --value)
now_us=$(awk '{printf "%.0f", $1 * 1000000}' /proc/uptime)
if [ -n "$active_us" ] && [ "$active_us" -gt 0 ] 2>/dev/null; then
    age=$(( (now_us - active_us) / 1000000 ))
    [ "$age" -ge "$warmup" ] || exit 0
fi

gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk 'NF {sum += $1; count++} END {if (count) printf "%.0f", sum / count}')
[ -n "$gpu_util" ] || exit 0
[ "$gpu_util" -le 1 ] || exit 0

health=$(curl -fsS --max-time 5 "$url/health" 2>/dev/null) || exit 0
printf '%s' "$health" | python3 -c '
import json, sys
s = json.load(sys.stdin).get("scheduler", {})
raise SystemExit(0 if s.get("active") == 0 and s.get("queued") == 0 else 1)
' || exit 0

payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":1,"think":false}' "$model")
if timeout "$probe_timeout" curl -fsS "$url/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1; then
    exit 0
fi

sleep 3
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk 'NF {sum += $1; count++} END {if (count) printf "%.0f", sum / count}')
[ -n "$gpu_util" ] || exit 0
[ "$gpu_util" -le 1 ] || exit 0

health=$(curl -fsS --max-time 5 "$url/health" 2>/dev/null) || exit 0
printf '%s' "$health" | python3 -c '
import json, sys
s = json.load(sys.stdin).get("scheduler", {})
raise SystemExit(0 if s.get("active") == 0 and s.get("queued") == 0 else 1)
' || exit 0

stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$state_dir/wedge-diag-$stamp.log"
pid=$(systemctl --user show "$service" -p MainPID --value)
if [ "$diagnostics" = 1 ]; then
    {
        echo "Colibri watchdog confirmed an idle inference failure at $stamp"
        echo "service=$service pid=$pid url=$url gpu_util=$gpu_util"
        echo
        systemctl --user status "$service" --no-pager || true
        echo
        nvidia-smi || true
        echo
        ss -tnp || true
        echo
        journalctl --user -u "$service" -n 200 --no-pager || true
        if command -v gdb >/dev/null 2>&1 && [ "${pid:-0}" -gt 1 ] 2>/dev/null; then
            echo
            timeout 20 gdb -q -batch -p "$pid" -ex 'thread apply all bt' || true
        fi
    } >"$log" 2>&1
fi

if [ "$dry_run" = 1 ]; then
    echo "Colibri watchdog would restart $service (diagnostics: $log)"
    exit 0
fi

echo "Colibri watchdog restarting $service after confirmed idle inference failure" >&2
systemctl --user restart "$service"
