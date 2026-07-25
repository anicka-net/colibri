#!/usr/bin/env bash
# Build and validate the TC-gather/FP8 candidate before running model gates.
set -euo pipefail

work=${COLI_GATE_WORK:-/home/claudia/colibri-tcg-candidate}
cuda_image=${COLI_CUDA_IMAGE:-nvidia/cuda:13.1.1-devel-ubuntu24.04}
pass_marker=${COLI_GATE_PASS_MARKER:-/home/claudia/colibri-tcg-fp8-harness.PASS}
gate=$work/tools/run_gb10_nvfp4_kv_gate.sh
uid=$(id -u)
gid=$(id -g)

run_docker() {
    local command
    printf -v command '%q ' docker "$@"
    /usr/bin/sg docker -c "$command"
}

rm -f "$pass_marker"
test -x "$gate"
command -v docker >/dev/null
run_docker image inspect "$cuda_image" >/dev/null

systemctl --user stop colibri-server.service colibri-watchdog.timer \
    >/dev/null 2>&1 || true
if pgrep -u "$uid" -f '(^|/)(colibri|glm)( |$)' >/dev/null 2>&1; then
    echo "another Colibri engine is running" >&2
    exit 2
fi

run_docker run --rm --gpus all \
    --user "$uid:$gid" \
    -e HOME=/tmp \
    -v "$work:$work" \
    -w "$work/c" \
    "$cuda_image" \
    bash "$work/tools/run_gb10_cuda_harness_in_container.sh"

touch "$pass_marker"
exec "$gate"
