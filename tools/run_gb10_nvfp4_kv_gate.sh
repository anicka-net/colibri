#!/usr/bin/env bash
# Focused long-context gate for NVFP4 resident formats and KV device shadows.
set -euo pipefail

work=${COLI_GATE_WORK:-/home/claudia/colibri-tcg-candidate}
binary=${COLI_GATE_BINARY:-$work/c/colibri}
driver=$work/tools/benchmark_service_throughput.py
nvroot=${COLI_GATE_NVROOT:-/home/claudia/models/SuperGLM-5.2-abliterated-NVFP4-506e95b9.aligned-v2-local}
result_root=${COLI_GATE_RESULTS:-/home/claudia/colibri-tcg-fp8-results}
max_tokens=${COLI_GATE_MAX_TOKENS:-64}
run_id=$(date -u +%Y%m%dT%H%M%SZ)
results=$result_root/$run_id
mkdir -p "$results/usage"

models=(
    "nvfp4-faithful|$nvroot/faithful|16"
    "nvfp4-compact|$nvroot/compact|40"
)

restore_usage() {
    local spec label model frozen
    for spec in "${models[@]}"; do
        IFS='|' read -r label model _ <<<"$spec"
        frozen=$results/usage/$label.coli_usage
        [[ ! -s "$frozen" ]] || cp -p "$frozen" "$model/.coli_usage"
    done
}

finish() {
    local rc=$?
    trap - EXIT
    restore_usage || true
    if ((rc == 0)); then
        touch "$results/COMPLETE"
    else
        touch "$results/FAILED"
    fi
    ln -sfn "$results" "$result_root/latest"
    exit "$rc"
}
trap finish EXIT

for spec in "${models[@]}"; do
    IFS='|' read -r label model _ <<<"$spec"
    [[ ! -s "$model/.coli_usage" ]] ||
        cp -p "$model/.coli_usage" "$results/usage/$label.coli_usage"
done

common_env=(
    TEMP=0
    CTX=16384
    COLI_CUDA=1
    CUDA_DENSE=1
    COLI_CUDA_HOST_EXPERTS=1
    CUDA_EXPERT_GB=0
    CUDA_RELEASE_HOST=0
    COLI_CUDA_PIPE=2
    COLI_CUDA_PIPE_S_MIN=1
    COLI_CUDA_PREFILL=0
    COLI_CUDA_TC_W4A16=1
    AUTOPIN=0
    DIRECT=1
    URING=1
    PIPE=1
    PIPE_WORKERS=10
    PILOT_REAL=0
    PILOT_K=6
    PILOT_TWO=0
    DRAFT=0
    CAP_RAISE=0
    CACHE_ROUTE=0
    COLI_ADAPTIVE_CAP=0
    COLI_DBG_DSACHAIN=1
    COLI_DSA_REFRESH=1
    COLI_KV_SHARE=0
    COLI_KV_CACHE_GB=0
    KVSAVE=0
    COLI_NVFP4_NATIVE=1
    COLI_NVFP4_NATIVE_MIN_ROWS=1
    COLI_SERVE_ALL_STOPS=0
    PROF=1
    OMP_NUM_THREADS=10
    OMP_PLACES=cores
    OMP_PROC_BIND=spread
)

{
    echo "run_id=$run_id"
    echo "host=$(hostname)"
    echo "commit=$(git -C "$work" rev-parse HEAD)"
    echo "binary=$binary"
    echo "max_tokens=$max_tokens"
    echo "workload=6009-token long programming prompt plus warmed short prompt"
    echo "matrix=faithful/compact x fp16/fp8"
} >"$results/metadata.txt"
printf 'label\tkv_dtype\tworkload\tstarted\tended\tseconds\n' >"$results/status.tsv"

run_one() {
    local label=$1 model=$2 cap=$3 dtype=$4
    local stem=$label-$dtype frozen started ended t0 t1
    frozen=$results/usage/$label.coli_usage
    [[ ! -s "$frozen" ]] || cp -p "$frozen" "$model/.coli_usage"
    started=$(date -Is)
    t0=$(date +%s)
    env "${common_env[@]}" COLI_KV_DTYPE="$dtype" \
        python3 "$driver" \
        --executable "$binary" \
        --model "$model" \
        --label "$label-$dtype" \
        --workload long \
        --cap "$cap" \
        --expert-bits 4 \
        --dense-bits 8 \
        --max-tokens "$max_tokens" \
        --warmups 0 \
        --samples 1 \
        >"$results/$stem.jsonl" \
        2>"$results/$stem.log"
    t1=$(date +%s)
    ended=$(date -Is)
    printf '%s\t%s\tlong\t%s\t%s\t%s\n' \
        "$label" "$dtype" "$started" "$ended" "$((t1-t0))" >>"$results/status.tsv"
    [[ ! -s "$frozen" ]] || cp -p "$frozen" "$model/.coli_usage"

    grep -Eq '\[PROF\] DSA decode TC gather: [1-9][0-9]* row \| 0 fallback' \
        "$results/$stem.log"
    grep -Eq '\[PROF\] DSA engagement: .*fallback pre 0 chain 0.*prefill full [1-9][0-9]* shared [1-9][0-9]* fallback 0' \
        "$results/$stem.log"
    grep -Eq '\[PROF\] resident dense layers: [1-9][0-9]* engaged \| 0 fallback' \
        "$results/$stem.log"
    grep -Eq '\[CUDA\] NVFP4: native [1-9][0-9]* \| generic 0 \| native-unavailable 0 \| failures 0' \
        "$results/$stem.log"
    if [[ "$dtype" == fp8 ]]; then
        grep -Eq '\[CUDA\] FP8 KV: quantized rows [1-9][0-9]* \| reader rows [1-9][0-9]* \| fallbacks 0' \
            "$results/$stem.log"
    fi
}

run_short() {
    local label=$1 model=$2 cap=$3 dtype=$4
    local stem=$label-$dtype-short frozen started ended t0 t1
    frozen=$results/usage/$label.coli_usage
    [[ ! -s "$frozen" ]] || cp -p "$frozen" "$model/.coli_usage"
    started=$(date -Is)
    t0=$(date +%s)
    env "${common_env[@]}" CTX=4096 COLI_KV_DTYPE="$dtype" \
        python3 "$driver" \
        --executable "$binary" \
        --model "$model" \
        --label "$label-$dtype" \
        --workload short \
        --cap "$cap" \
        --expert-bits 4 \
        --dense-bits 8 \
        --max-tokens "$max_tokens" \
        --warmups 1 \
        --samples 2 \
        >"$results/$stem.jsonl" \
        2>"$results/$stem.log"
    t1=$(date +%s)
    ended=$(date -Is)
    printf '%s\t%s\tshort\t%s\t%s\t%s\n' \
        "$label" "$dtype" "$started" "$ended" "$((t1-t0))" >>"$results/status.tsv"
    [[ ! -s "$frozen" ]] || cp -p "$frozen" "$model/.coli_usage"
    grep -Eq '\[PROF\] resident dense layers: [1-9][0-9]* engaged \| 0 fallback' \
        "$results/$stem.log"
    grep -Eq '\[CUDA\] NVFP4: native [1-9][0-9]* \| generic 0 \| native-unavailable 0 \| failures 0' \
        "$results/$stem.log"
    if [[ "$dtype" == fp8 ]]; then
        grep -Eq '\[CUDA\] FP8 KV: quantized rows [1-9][0-9]* \| reader rows [1-9][0-9]* \| fallbacks 0' \
            "$results/$stem.log"
    fi
}

systemctl --user stop colibri-server.service colibri-watchdog.timer \
    >/dev/null 2>&1 || true
for spec in "${models[@]}"; do
    IFS='|' read -r label model cap <<<"$spec"
    run_one "$label" "$model" "$cap" fp16
    run_one "$label" "$model" "$cap" fp8
done
if [[ ${COLI_GATE_SKIP_SHORT:-0} != 1 ]]; then
    for spec in "${models[@]}"; do
        IFS='|' read -r label model cap <<<"$spec"
        run_short "$label" "$model" "$cap" fp16
        run_short "$label" "$model" "$cap" fp8
    done
fi
