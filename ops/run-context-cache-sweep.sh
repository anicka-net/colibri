#!/usr/bin/env bash
# Run the persisted-KV context/cache sweep while production is offline, then
# restore any pre-existing checkpoint and restart production even on failure.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 REPO MODEL PYTHON RESULT_DIR" >&2
    exit 2
fi

repo=$1
model=$2
python=$3
result_dir=$4
service_env=$HOME/.config/colibri/service.env
kv_file=$model/.coli_kv
saved_kv=$result_dir/preexisting.coli_kv
benchmark_kv=$result_dir/benchmark.coli_kv
had_kv=0

mkdir -p "$result_dir"
if [[ -e "$saved_kv" || -e "$benchmark_kv" ]]; then
    echo "refusing to overwrite an existing sweep checkpoint in $result_dir" >&2
    exit 2
fi
if [[ -e "$kv_file" ]]; then
    mv "$kv_file" "$saved_kv"
    had_kv=1
fi

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -e "$kv_file" ]]; then
        mv "$kv_file" "$benchmark_kv"
    fi
    if [[ $had_kv -eq 1 ]]; then
        mv "$saved_kv" "$kv_file"
    fi
    systemctl --user start colibri-server.service
    systemctl --user start colibri-watchdog.timer
    exit "$status"
}
trap cleanup EXIT INT TERM

systemctl --user stop colibri-watchdog.timer colibri-watchdog.service
systemctl --user stop colibri-server.service

set -a
# shellcheck source=/dev/null
source "$service_env"
set +a

"$python" "$repo/tools/benchmark_context_cache.py" \
    --executable "$repo/c/colibri" \
    --model "$model" \
    --target-prompt-tokens 32000 \
    --max-tokens 256 \
    --contexts 131072 98304 65536 32768
