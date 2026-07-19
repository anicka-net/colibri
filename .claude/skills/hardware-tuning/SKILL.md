---
name: hardware-tuning
description: How to configure and run the colibri GLM engine optimally on each hardware class — discrete multi-GPU with large RAM, unified-memory SoC, or CPU-only with limited RAM. Use when setting up a new machine, tuning throughput, or choosing env vars for a benchmark or service.
---

# Running colibri on different hardware

Pick the profile that matches the machine, then apply the shared notes at the end.
Full env-var reference: `docs/ENVIRONMENT.md`. Numbers below are for a ~750B-class
MoE (78 layers, 256 experts/layer, ~19 MB/expert int4); scale expectations accordingly.

## Profile A — discrete GPU(s) + RAM larger than the model

The model lives wired in RAM; the GPUs hold a hot expert tier, all dense weights,
and the resident decode pipeline. This is the fastest profile: **19.6 tok/s**
greedy short-context on 2× H100 NVL (`MTP=0`), **17.9 tok/s** with `DRAFT=1`
at 91% acceptance.

```
COLI_CUDA=1 COLI_GPUS=0,1 CUDA_DENSE=1 \
CUDA_EXPERT_GB=<see below> \
COLI_CUDA_PIPE=2 COLI_CUDA_PIPE_S_MIN=1 \
COLI_CUDA_TC_W4A16=1 \
OMP_NUM_THREADS=<physical cores> OMP_PLACES=cores OMP_PROC_BIND=spread \
DIRECT=0 ./glm 256 4 4
```

- Put the snapshot on tmpfs (or rely on the page cache) and use `DIRECT=0`:
  with everything RAM-resident there is no NVMe path to protect.
- `CUDA_EXPERT_GB`: total across GPUs. Fill VRAM but leave 3–9 GB per GPU for
  dense weights, KV cache growth, and scratch. Past the point where the CPU-side
  expert work and GPU expert work are balanced, more budget stops helping —
  sweep in 5–10 GB steps and keep the knee.
- `COLI_CUDA_PIPE=2` + `COLI_CUDA_PIPE_S_MIN=1` enables the resident decode
  stream with the fused attention chain (S<=4). On multi-GPU, layers get
  contiguous home-device blocks, so the residual crosses P2P once per forward.
- `COLI_CUDA_TC_W4A16=1` is safe to keep on always: it only engages at rows>=16
  (prefill/batch), never decode, and the weights path is lossless.
- Speculative decoding: with an **int8** MTP head present, `DRAFT=1` reaches
  ~90% acceptance. At short context it is roughly break-even with `MTP=0`;
  at long context it wins. `DRAFT>1` degrades (self-fed drafts).
- `COLI_NUMA=1` measured neutral on decode (GPU-bound) — not worth sweeping.

### Long context on Profile A: DSA sparse attention

With the `out-idx-*` indexer files present the engine auto-enables DSA, and as
of inc.4 (2026-07-19) it is a **strict win past the ~4k crossover**: at 6.7k
context, prefill 392 s vs 411 s dense and decode 3.8 vs 3.2 tok/s (+20%), with
decode attention ~flat in T while dense grows linearly. Below index_topk (2048)
DSA-on costs nothing (the chain caches k_idx itself). Nothing to configure —
the defaults do this. Knobs and caveats:

- `COLI_DSA_CHAIN=0` / `COLI_DSA_TCGATHER=0` fall back to the slower
  CPU-selection / scalar sel-absorb paths (debug/A-B only).
- `COLI_DBG_DSACHAIN=1` prints engagement counters
  (`[DSAC] engaged: ... fallback: ...`) — check `fb 0` after any change in this
  area; the fallbacks are silent by design.
- Current ceiling: the device KV/Ic shadows and absorb kernels cap at
  **T<=8192**; past that the engine falls back to CPU paths (cap lift is the
  next queue item).
- MTP composes with DSA (84–88% acceptance at 2.8–6.7k) but adds only ~5%
  end-to-end until small-S GPU forwards get cheaper.

## Profile B — unified-memory SoC (Grace-class, "Spark")

CPU and GPU share LPDDR; the GPU reads weights zero-copy, no VRAM tier or
upload path. RAM is smaller than the model, so experts stream from NVMe and the
expert cache hit rate — not compute — decides throughput. Two distinct regimes:

**Short-context (CTX<=4096) benchmark/interactive profile** — no separate CUDA
tier, big LRU:

```
COLI_CUDA=1 COLI_CUDA_HOST_EXPERTS=1 COLI_CUDA_PIPE=2 \
CUDA_EXPERT_GB=0 PIN=0 PIN_GB=0 DIRECT=1 AUTOPIN=0 \
PILOT_REAL=1 COUPLE_K=8 COUPLE_D=1 \
OMP_NUM_THREADS=<cores> CTX=4096 ./glm 63 4 4
```

Strict top-8: **3.58 tok/s** (coupling prefetch raises hit 87.8→92.3%; without
it 3.27). `CACHE_ROUTE` J2/M12 reaches 4.74 tok/s at 96.3% hit but substitutes
12.9% of route slots — label results from it as a quality-changing mode.

**131k production/service profile** — 30 GB CUDA tier + small LRU:

```
... CUDA_EXPERT_GB=30 COLI_ADAPTIVE_CAP=1 PILOT_K=6 CTX=131072 ./glm 17 4 4
```

- `COLI_ADAPTIVE_CAP=1` (deployed on the mux service): borrows the untouched
  full-context KV reservation for extra cache slots at request boundaries —
  **1.21 → 1.43 tok/s**, hit 76→84%, no swap. Requires `KV_SLOTS=1`,
  `COLI_CUDA_PIPE=0`, non-mmap serve.
- Keep coupling prefetch OFF here and `PILOT_K=6` — both boundaries are
  measured knees (see PERF-QUEUE "measured dead ends" before re-sweeping).
- Never combine the production CUDA tier with a large cache cap: same physical
  LPDDR, forces swap.
- Prefill: `COLI_CUDA_PREFILL=1` (pipe0 CUDA prefill, no full-context device
  shadows) + W4A16: 86 → 64 s on a frozen 260-token prompt; with DSA past
  index_topk the transient-Ic device path is 2.66× the CPU-DSA path and ~29%
  faster than dense at 2.8k.
- Speculation: keep `DRAFT=0` in strict mode — miss-containing host groups
  synchronize before slab reuse and it measures a net loss (3.27 → 1.65).
- The theoretical ceiling is NVMe bandwidth / (misses/token × expert size);
  prefetch/eviction policy work pays more than kernel work on this profile.

## Profile C — CPU-only, limited RAM

No CUDA. Same streaming regime as Profile B but all matmuls on cores (int8/int4
IDOT paths, AVX2/NEON). Expect single-digit tok/s at best.

```
OMP_NUM_THREADS=<physical cores> OMP_PLACES=cores OMP_PROC_BIND=spread \
DIRECT=1 DRAFT=0 ./glm <cache-slots> 4 4
```

- `OMP_PLACES=cores OMP_PROC_BIND=spread` matters most here (up to ~2x over
  unpinned on high-core-count parts).
- `DRAFT=0`: verify forwards multiply CPU attention cost; speculation does not
  pay without a fast small-S path.
- Cache slots: (free RAM − dense-resident GB) / expert size. When RAM is very
  tight, prefer fewer slots over swapping — a wired model plus swap thrashing
  is worse than honest streaming.

## Prefill (long prompts)

Prefill is a separate performance regime from decode. On discrete-GPU hosts the
tensor-core prefill paths are default-on (`COLI_PREFILL_GEMM`, `COLI_PIPE_TC`)
except `COLI_CUDA_TC_W4A16=1` which you should add explicitly (lossless, halves
expert prefill time). Measured on the 2.7k prompt, compounding:
score-softmax-value GEMM rewrite 161.7 → 80.4 s, pipe_gemm TC 78.4 → 54.0 s,
and with DSA selection active 51.7 s (inc.4 TC gather; was 267.8 before the
DSA prefill work). At 6.7k: 392 s DSA vs 411 s dense. Do NOT enable
`COLI_CUDA_TC_INT4` (W4A4) — slower and lossy. These paths use fp16 tensor-core
inputs (fp32 accumulate), so greedy continuations can differ from the fp32
kernels within run-to-run variance; set them to 0 for bit-level comparisons.
On Spark use `COLI_CUDA_PREFILL=1` instead (see Profile B).

## Shared notes

- **MTP head precision**: the speculative head must be converted at **int8**
  (`convert_fp8_to_int4.py --mtp`, which forces it). An int4 MTP head gives
  ~0% acceptance and the engine auto-disables drafts after 24 misses.
- **Benchmarking**: use `TEMP=0` for determinism. For config A/B comparisons,
  freeze the expert-placement state: run 2 unfrozen warmups on the target
  prompt class, snapshot `.coli_usage`, and restore it before every measured
  run (the file evolves each run and moves results by a few percent — and a
  degenerate run can poison it for long prompts). Identical frozen state is
  bit-reproducible since the softmax-race fix.
- **Prefill vs decode**: prefill is reported separately; judge configs on the
  decode tok/s line. Cold first runs are disk-bound and not representative.
- **Hit-rate telemetry**: on discrete GPUs the headline number is the VRAM-served
  share; on unified/CPU it is the RAM (disk-miss) hit rate. `PROF=1` gives the
  phase breakdown when something is slower than expected.

## Shared-expert placement (COLI_FUSE_SHARED)

By default the shared expert is dispatched async on the GPU right before the
CPU enters `moe()`, so the two overlap. `COLI_FUSE_SHARED=1` instead folds it
into the fused attention chain (before the chain's sync). Measured on a
high-core-count discrete-GPU host: overlap wins (17.0 vs 16.5 tok/s) — fusing
delays the downloads that unblock the CPU. On machines with weak CPUs (where
`moe()` routing is slow relative to the GPU), try `=1` and measure.
