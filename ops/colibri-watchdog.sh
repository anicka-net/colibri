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
api_key=${COLI_WATCHDOG_API_KEY:-${COLI_API_KEY:-}}
cooldown=${COLI_WATCHDOG_COOLDOWN_SECONDS:-600}

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

gpu_util=$(timeout 10 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk 'NF {sum += $1; count++} END {if (count) printf "%.0f", sum / count}')
[ -n "$gpu_util" ] || exit 0
[ "$gpu_util" -le 1 ] || exit 0

health=$(curl -fsS --max-time 5 "$url/health" 2>/dev/null) || exit 0
printf '%s' "$health" | grep -q '"scheduler":{"active":0,"queued":0' || exit 0

payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":1,"think":false}' "$model")
auth=()
[ -z "$api_key" ] || auth=(-H "Authorization: Bearer $api_key")
set +e
timeout "$probe_timeout" curl -fsS "$url/v1/chat/completions" \
    -H 'Content-Type: application/json' -H 'X-Colibri-Watchdog: 1' \
    "${auth[@]}" -d "$payload" >/dev/null 2>&1
probe_status=$?
set -e
if [ "$probe_status" -eq 0 ]; then
    exit 0
fi
[ "$probe_status" -eq 124 ] || [ "$probe_status" -eq 137 ] || exit 0

sleep 3
gpu_util=$(timeout 10 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk 'NF {sum += $1; count++} END {if (count) printf "%.0f", sum / count}')
[ -n "$gpu_util" ] || exit 0
[ "$gpu_util" -le 1 ] || exit 0

health=$(curl -fsS --max-time 5 "$url/health" 2>/dev/null) || exit 0
if ! printf '%s' "$health" | grep -q '"scheduler":{"active":0,"queued":0'; then
    printf '%s' "$health" | grep -q '"scheduler":{"active":1,"queued":0.*"watchdog_active":1' || exit 0
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$state_dir/wedge-diag-$stamp.log"
pid=$(systemctl --user show "$service" -p MainPID --value)
last_restart="$state_dir/watchdog-last-restart"
if [ -f "$last_restart" ]; then
    last=$(cat "$last_restart" 2>/dev/null || echo 0)
    now=$(date +%s)
    [ $((now - last)) -ge "$cooldown" ] || exit 0
fi
if [ "$diagnostics" = 1 ]; then
    {
        echo "Colibri watchdog confirmed an idle inference failure at $stamp"
        echo "service=$service pid=$pid url=$url gpu_util=$gpu_util"
        echo
        timeout 10 systemctl --user status "$service" --no-pager || true
        echo
        timeout 10 nvidia-smi || true
        echo
        timeout 10 ss -tnp || true
        echo
        timeout 10 journalctl --user -u "$service" -n 200 --no-pager || true
        if command -v gdb >/dev/null 2>&1 && [ "${pid:-0}" -gt 1 ] 2>/dev/null; then
            echo
            timeout 20 gdb -q -batch -p "$pid" -ex 'thread apply all bt' || true
        fi
    } >"$log" 2>&1
    find "$state_dir" -maxdepth 1 -name 'wedge-diag-*.log' -mtime +7 -delete
    ls -1t "$state_dir"/wedge-diag-*.log 2>/dev/null | tail -n +21 | xargs -r rm --
fi

if [ "$dry_run" = 1 ]; then
    echo "Colibri watchdog would restart $service (diagnostics: $log)"
    exit 0
fi

echo "Colibri watchdog restarting $service after confirmed idle inference failure" >&2
date +%s >"$last_restart"
systemctl --user restart "$service"
