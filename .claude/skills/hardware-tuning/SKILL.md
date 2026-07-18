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
and the resident decode pipeline. This is the fastest profile (order of 15+ tok/s).

```
COLI_CUDA=1 COLI_GPUS=0,1 CUDA_DENSE=1 \
CUDA_EXPERT_GB=<see below> \
COLI_CUDA_PIPE=2 COLI_CUDA_PIPE_S_MIN=1 \
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
- Speculative decoding: with an **int8** MTP head present, `DRAFT=1` reaches
  ~90% acceptance. At short context it is roughly break-even with `MTP=0`;
  at long context it wins. `DRAFT>1` degrades (self-fed drafts).
- Two-socket hosts: try `COLI_NUMA=1` (interleaves resident weights).

## Profile B — unified-memory SoC (Grace-class, "Spark")

CPU and GPU share LPDDR; the GPU reads weights zero-copy, no VRAM tier or
upload path. RAM is smaller than the model, so experts stream from NVMe and the
expert cache hit rate — not compute — decides throughput.

```
COLI_CUDA=1 DIRECT=1 AUTOPIN=0 \
OMP_NUM_THREADS=<cores> ./glm 64 4 4
```

- `DIRECT=1` (O_DIRECT): buffered readahead wastes NVMe bandwidth on this path.
- Cache size (first CLI arg) as large as RAM allows after the dense weights;
  every extra slot is hit rate. Steady-state LRU hit rate on a fresh long
  corpus is ~77% — quote that, not the flattering just-after-prefill number.
- Speculation helps MORE here than on Profile A: drafted tokens share expert
  loads, and disk is the bottleneck. Use `DRAFT=1` with an int8 MTP head.
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

Prefill is a separate performance regime from decode.  On GPU hosts always add
`COLI_CUDA_TC_W4A16=1`: the tensor-core expert path (lossless weights, fp16
activations, rows>=16) halves prefill expert time — measured 207 -> 167 s on a
2.7k prompt, output identical.  Do NOT enable `COLI_CUDA_TC_INT4` (W4A4) — it
is slower and lossy.  Prefill attention is currently the dominant cost at long
prompts (see docs/PERF-QUEUE.md); expect ~16 tok/s prefill until the tiled
kernel lands.

## Shared notes

- **MTP head precision**: the speculative head must be converted at **int8**
  (`convert_fp8_to_int4.py --mtp`, which forces it). An int4 MTP head gives
  ~0% acceptance and the engine auto-disables drafts after 24 misses.
- **Benchmarking**: use `TEMP=0` for determinism, and expect a few percent
  run-to-run drift anyway — the expert pin-placement stats file updates each
  run. Compare configs on the same run count and prompt.
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
