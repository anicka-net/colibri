# GB10 NVFP4 validation gate

The native backend is compile-validated on Twilight with CUDA 13.3 and targets
`sm_121a`. Deployment remains pinned to the CUDA 13.1 development container;
repeat every build below there before using a converted snapshot.

## Build and numerical gates

From `c/`:

```sh
make -B glm CUDA=1 NVFP4_NATIVE=1 CUDA_ARCH=sm_121a
make cuda-nvfp4-layout-test CUDA_ARCH=sm_121a
make cuda-nvfp4-test CUDA_ARCH=sm_121a
make cuda-test CUDA_ARCH=sm_121a NVFP4_NATIVE=1
```

`cuda-nvfp4-test` covers both production expert projections, `[S,6144] x
[2048,6144]^T` and `[S,2048] x [6144,2048]^T`, for row counts 1, 2, 8, 16,
and 64. It compares sampled outputs with the software W4A4 oracle and requires
10 native engagements with zero generic, unavailable, or failure counts.
The ordinary CUDA harness additionally checks single-GEMM and routed-group
NVFP4 parity. Exit 77 means the machine has no supported NVIDIA device and is
a skip, not a pass.

Run the generic control explicitly:

```sh
COLI_NVFP4_NATIVE=0 ./backend_cuda_test
```

The engine accepts `COLI_NVFP4_NATIVE_MIN_ROWS=N`, allowing S=1 decode to stay
on W4A32 while larger prefill batches use CUTLASS. Do not choose a threshold
until both projection shapes have been measured on GB10. Record winning and
rejected thresholds in `docs/PERF-QUEUE.md`.

## Snapshot and service gates

1. Run the converter with an exact source revision, then verify faithful and
   compact expert payloads have the same inode and link count.
2. Start with the faithful BF16-resident snapshot only. Run `coli plan` and
   record the measured memory envelope before creating or enabling compact.
3. Confirm startup accepts the versioned manifest and every expert reports the
   CUTLASS SM1xx scale layout. Any malformed metadata must fail loading.
4. Exercise prefill and decode separately. Report context, rows per expert,
   native/generic calls, every fallback reason, cache hit rate, expert bytes,
   felt I/O wait, and KV memory.
5. Require native prefill to beat the generic path. Select S=1 by measurement;
   generic decode is acceptable when faster.
6. Run expert eviction/reload and host-backed execution before quality tests.
7. Keep `COLI_KV_DTYPE=fp8` opt-in through rewind, prefix restore, DSA
   FULL/SHARED, and 4k/32k validation. Require no invalid values and no FP8
   shadow fallback.
8. Freeze `.coli_usage`, use `TEMP=0`, and run INT4, faithful, and compact
   quality rungs. Faithful must improve aggregate log-likelihood without a
   deterministic-suite regression. Compact must remain within 0.02 nat/token
   and introduce no structural failures.

Do not deploy to pondermatic until its existing RAM workload is confirmed safe
to coexist or stop. Do not place the snapshot on deepthought.
