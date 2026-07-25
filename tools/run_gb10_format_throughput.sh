#!/usr/bin/env bash
# Matched production-oriented throughput matrix for one GB10 Spark.
set -uo pipefail

work=${COLI_BENCH_WORK:-/home/claudia/colibri-throughput}
binary=${COLI_BENCH_BINARY:-$work/c/colibri}
driver=$work/tools/benchmark_service_throughput.py
int4=${COLI_BENCH_INT4:-/home/claudia/moe-cache-test/GLM-5.2-colibri-int4}
nvroot=${COLI_BENCH_NVROOT:-/home/claudia/models/SuperGLM-5.2-abliterated-NVFP4-506e95b9.aligned-v2-local}
result_root=${COLI_BENCH_RESULTS:-/home/claudia/colibri-throughput-results}
kv_dtype=${COLI_BENCH_KV_DTYPE:-fp16}
run_id=$(date -u +%Y%m%dT%H%M%SZ)
results=$result_root/$run_id
mkdir -p "$results/usage"

models=(
    # label | snapshot | dense bits
    "int4|$int4|4"
    "nvfp4-faithful|$nvroot/faithful|8"
    "nvfp4-compact|$nvroot/compact|8"
)

restore_usage() {
    local label model frozen
    for spec in "${models[@]}"; do
        IFS='|' read -r label model _ <<<"$spec"
        frozen=$results/usage/$label.coli_usage
        if [[ -s "$frozen" ]]; then
            cp -p "$frozen" "$model/.coli_usage"
        fi
    done
}
trap restore_usage EXIT

for spec in "${models[@]}"; do
    IFS='|' read -r label model _ <<<"$spec"
    if [[ -s "$model/.coli_usage" ]]; then
        cp -p "$model/.coli_usage" "$results/usage/$label.coli_usage"
    fi
done

{
    echo "run_id=$run_id"
    echo "host=$(hostname)"
    echo "commit=$(git -C "$work" rev-parse HEAD)"
    echo "binary=$binary"
    echo "driver=$driver"
    echo "kv_dtype=$kv_dtype"
    echo "temperature=0"
    echo "comparison=production-oriented format-specific strict-top8 profiles"
    echo "short_context=4096"
    echo "short_int4=cap63_cuda0_pipe2_pilot6"
    echo "short_faithful=cap16_cuda0_pipe2_no-pilot"
    echo "short_compact=cap40_cuda0_pipe2_no-pilot"
    echo "long_context=16384"
    echo "long_int4=cap17_cuda30_pin30_pipe0_adaptive_pilot6"
    echo "long_faithful=cap16_cuda0_pipe2_no-pilot"
    echo "long_compact=cap40_cuda0_pipe2_no-pilot"
    echo "requested caps are subject to the runtime RSS/resource guard"
    echo "short=2_warmups_3_samples_64_output_tokens"
    echo "long=2_warmups_2_samples_64_output_tokens"
    echo "long_prompt_is_a_6009-token_programming_protocol_and_code_review"
} >"$results/metadata.txt"
{
    date -Is
    uname -a
    lscpu
    free -h
    nvidia-smi
    findmnt -T "$int4"
} >"$results/hardware.txt" 2>&1

common_env=(
    TEMP=0
    COLI_CUDA=1
    CUDA_DENSE=1
    COLI_CUDA_HOST_EXPERTS=1
    COLI_CUDA_TC_W4A16=1
    AUTOPIN=0
    DIRECT=1
    URING=1
    PIPE=1
    PIPE_WORKERS=10
    PILOT_TWO=0
    DRAFT=0
    CAP_RAISE=0
    CACHE_ROUTE=0
    COLI_DBG_DSACHAIN=1
    COLI_DSA_REFRESH=1
    COLI_KV_DTYPE="$kv_dtype"
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
printf '%s\n' "${common_env[@]}" >"$results/environment.txt"
printf 'label\tworkload\tstarted\tended\tseconds\texit\n' >"$results/status.tsv"
failures=0

run_one() {
    local label=$1 model=$2 dense_bits=$3 workload=$4 warmups=$5 samples=$6
    local cap ctx cuda_gb cuda_pipe cuda_prefill pilot_real
    local pilot_k adaptive_cap release_host
    local started ended t0 t1 rc frozen
    local -a pin_env=()

    if [[ "$workload" == short ]]; then
        ctx=4096
        cuda_gb=0
        release_host=0
        cuda_pipe=2
        cuda_prefill=0
        adaptive_cap=0
        if [[ "$label" == int4 ]]; then
            cap=63
            pilot_real=1
            pilot_k=6
        elif [[ "$label" == nvfp4-faithful ]]; then
            cap=16
            pilot_real=0
            pilot_k=6
        else
            cap=40
            pilot_real=0
            pilot_k=6
        fi
    else
        ctx=16384
        if [[ "$label" == int4 ]]; then
            cap=17
            cuda_gb=30
            release_host=1
            pin_env=(PIN=auto PIN_GB=30)
            cuda_pipe=0
            cuda_prefill=1
            adaptive_cap=1
            pilot_real=1
            pilot_k=6
        elif [[ "$label" == nvfp4-faithful ]]; then
            cap=16
            cuda_gb=0
            release_host=0
            cuda_pipe=2
            cuda_prefill=0
            adaptive_cap=0
            pilot_real=0
            pilot_k=6
        else
            cap=40
            cuda_gb=0
            release_host=0
            cuda_pipe=2
            cuda_prefill=0
            adaptive_cap=0
            pilot_real=0
            pilot_k=6
        fi
    fi

    frozen=$results/usage/$label.coli_usage
    [[ ! -s "$frozen" ]] || cp -p "$frozen" "$model/.coli_usage"
    started=$(date -Is)
    t0=$(date +%s)
    env -u PIN -u PIN_GB "${common_env[@]}" \
        "${pin_env[@]}" \
        CUDA_EXPERT_GB="$cuda_gb" \
        CUDA_RELEASE_HOST="$release_host" \
        COLI_CUDA_PIPE="$cuda_pipe" \
        COLI_CUDA_PIPE_S_MIN=1 \
        COLI_CUDA_PREFILL="$cuda_prefill" \
        PILOT_REAL="$pilot_real" \
        PILOT_K="$pilot_k" \
        COLI_ADAPTIVE_CAP="$adaptive_cap" \
        CTX="$ctx" \
        python3 "$driver" \
        --executable "$binary" \
        --model "$model" \
        --label "$label" \
        --workload "$workload" \
        --cap "$cap" \
        --expert-bits 4 \
        --dense-bits "$dense_bits" \
        --max-tokens 64 \
        --warmups "$warmups" \
        --samples "$samples" \
        >"$results/$label-$workload.jsonl" \
        2>"$results/$label-$workload.log"
    rc=$?
    t1=$(date +%s)
    ended=$(date -Is)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$workload" "$started" "$ended" "$((t1-t0))" "$rc" \
        >>"$results/status.tsv"
    [[ ! -s "$frozen" ]] || cp -p "$frozen" "$model/.coli_usage"
    if ((rc != 0)); then
        failures=$((failures+1))
        touch "$results/FAILED"
    fi
    return 0
}

systemctl --user stop colibri-server.service colibri-watchdog.timer \
    >/dev/null 2>&1 || true
if pgrep -u "$(id -u)" -f '(^|/)(colibri|glm)( |$)' >/dev/null 2>&1; then
    echo "another Colibri engine is running" >"$results/FAILED"
    exit 2
fi

# Complete the inexpensive workload first so useful deployment data arrives
# even if a later long-prefill run fails.
for spec in "${models[@]}"; do
    IFS='|' read -r label model dense_bits <<<"$spec"
    if [[ ${COLI_BENCH_SKIP_SHORT:-0} == 1 ]]; then
        continue
    fi
    if [[ "$label" == int4 && ${COLI_BENCH_SKIP_INT4_SHORT:-0} == 1 ]]; then
        continue
    fi
    run_one "$label" "$model" "$dense_bits" short 2 3
done

# Reverse the order to avoid always giving compact the final/order position.
for spec in "${models[2]}" "${models[1]}" "${models[0]}"; do
    IFS='|' read -r label model dense_bits <<<"$spec"
    if [[ ${COLI_BENCH_SKIP_LONG:-0} == 1 ]]; then
        continue
    fi
    if [[ "$label" == int4 && ${COLI_BENCH_SKIP_INT4_LONG:-0} == 1 ]]; then
        continue
    fi
    if [[ "$label" == nvfp4-compact && ${COLI_BENCH_SKIP_COMPACT_LONG:-0} == 1 ]]; then
        continue
    fi
    if [[ "$label" == nvfp4-faithful && ${COLI_BENCH_SKIP_FAITHFUL_LONG:-0} == 1 ]]; then
        continue
    fi
    run_one "$label" "$model" "$dense_bits" long 2 2
done

restore_usage
if ((failures == 0)); then
    touch "$results/COMPLETE"
fi
ln -sfn "$results" "$result_root/latest"
exit "$((failures != 0))"
