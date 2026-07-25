#!/usr/bin/env bash
# Invoked inside the pinned CUDA development container by the GB10 sequence.
set -euo pipefail

build=(
    CUDA=1
    NVFP4_NATIVE=1
    CUDA_ARCH=sm_121a
    CUDA_HOME=/usr/local/cuda
)

make -B gpu-compile "${build[@]}"
for dtype in fp32 fp16 fp8; do
    echo "RUN_KV_${dtype}"
    env COLI_KV_DTYPE="$dtype" ./backend_cuda_test
done

echo RUN_GENERIC
env COLI_KV_DTYPE=fp16 COLI_NVFP4_NATIVE=0 ./backend_cuda_test
make -B glm "${build[@]}" -j10
