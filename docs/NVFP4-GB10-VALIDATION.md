# GB10 NVFP4 validation gate

The native backend is compile-validated on Twilight with CUDA 13.3 and
correctness-validated on pondermatic's GB10 with CUDA 13.0.88, targeting
`sm_121a`. Deployment remains pinned to the CUDA 13.1 development container;
repeat every build below there before using a converted snapshot.

## Reconciled merge candidate

Candidate `83fc0db` includes `origin/main` and JustVugg `dev` through
`5724dae`. On Twilight it passes the full C suite and Python/API suite (221
passed, 13 skipped), the native-server suite (26 passed), and a CUDA 13.3 `sm_121a`
release build. The CUTLASS layout parity test passes. The ordinary CUDA
harness and dedicated NVFP4 oracle both compile; execution exits 77 because
Twilight has no NVIDIA device. The reconciled Inkling CUDA target also
compiles for `sm_121a`.

The remaining merge gates require a GB10: rerun both CUDA harnesses in the
pinned CUDA 13.1 container, run native and forced-generic full-model smokes,
repeat the matched native-prefill performance comparison, and run the
deterministic faithful/compact quality and tool suites on this exact commit.
FP8 KV remains opt-in until its 4k/32k device-shadow gate completes.

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
and 64. It compares sampled outputs with the software W4A4 oracle, then checks
two-expert grouped gate/up/down dispatch at production dimensions against
independent native expert MLPs for the same row counts. The gate requires 70
native problem engagements, 15 grouped launches covering 30 problems, and
zero generic, unavailable, grouped-fallback, or failure counts. The ordinary
CUDA harness additionally checks single-GEMM and routed-group NVFP4 parity.
Exit 77 means the machine has no supported NVIDIA device and is a skip, not a
pass.

Run the generic control explicitly:

```sh
COLI_NVFP4_NATIVE=0 ./backend_cuda_test
```

The engine accepts `COLI_NVFP4_NATIVE_MIN_ROWS=N`, allowing S=1 decode to stay
on W4A32 while larger prefill batches use CUTLASS. GB10 measurement selected
the default threshold of 1; the rejected threshold is 2 because native S=1 is
already more than twice as fast for both production projections.

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

Do not place the snapshot on deepthought.
