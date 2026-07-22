# Changelog

All notable changes to colibrì are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.1.0] — 2026-07-22

A community release. 27 pull requests from more than 20 contributors, 216 commits since
v1.0.0. Most of what follows was found, measured, or fixed by people who do not work on
this project and had nothing to gain from it.

### Added

- **AMD GPU support (HIP/ROCm)** (#339) — single-source `backend_gpu_compat.h` with a WMMA
  dispatch gate, so one codebase builds for CUDA and HIP. Validated on an RX 9070 XT
  (RDNA4, ROCm 7.2): token-exact against CPU on a real fmt=4 gs64 container, with resident
  dense *and* with routed experts in VRAM, plus a fail-injection control proving the GPU
  actually executed the work.
- **Dual-SSD streaming** (`COLI_MODEL_MIRROR`, #421) — read the model from two drives at
  once, roughly doubling streaming bandwidth on a disk-bound host.
- **N-drive shard split** (`COLI_MODEL_DIRS`, #469) — capacity aggregation: run a container
  no single drive can hold, spread across several with no duplication.
- **fmt=5 (int3-g64)** (#168) — 3-bit weights with per-64 group scales: measured 3.3x lower
  outlier-row error than per-row int4 at 25% fewer bytes.
- **fmt=6 (E8/IQ3 lattice)** (#465) — CPU decode kernel and dispatch; index codec tooling (#458).
- **`tools/try_tool_calling.py`** — dependency-free two-turn tool-calling probe that doubles
  as a smoke test.

### Fixed

- **Tool calling in coding clients** (#401), root cause found, in two parts:
  - **#506** — the engine capped prompt encoding at `CTX-2`, and the tokenizer stops dead at
    its limit *without reporting anything*. A prompt longer than the context was therefore
    silently truncated to its first `CTX-2` tokens and answered anyway. With the 4096 default
    that is 4094 — exactly the `prefill 4094` in the field report. The dropped tail was the
    tool instructions and the user's actual turn, so the model emitted a bare `<` and stopped;
    and because clients append to the *end* while truncation keeps the *head*, every retry
    re-sent a byte-identical prompt. Now refused with a 400 `context_length_exceeded`.
  - **#505** — a tool call whose closing `</tool_call>` never arrived was dropped whole,
    because the parser required both tags. Now recovered when unambiguous, on both the
    streamed and non-streamed paths.
  - **#437** — non-EOS role markers were armed as hard stops in serve mode and cut generation
    the instant a tool block started.
- **Grouped-int4 (fmt=4) produced garbage output on CUDA** with `CUDA_DENSE=1` (#298) — the
  dense and attention kernels applied per-group scales as if they were per-row. Hardware-verified.
- **OpenMP tuning re-exec preserved the CPU affinity mask** (#476), jailing every thread onto
  one core when `OMP_PROC_BIND`/`OMP_PLACES` were set: roughly a 20x slowdown.
- **Pilot eviction guard dropped ~100% of speculations** once the cache filled (#497),
  collapsing `PILOT_REAL` to a hint-only path.
- **Silent budget clamp** capped the CUDA expert tier at ~109 experts regardless of
  `CUDA_EXPERT_GB` (#495).
- fmt=4 guard at the per-row-only CUDA entry points (#464/#470); `COLI_CUDA_MTP=1` and
  `COLI_CUDA=0` are now honoured over implicit defaults (#468).

### Security

Threat model: model files come from mirrors that are not trusted.

- **#368** — server hardening, JSON and tokenizer parser hardening, build flags, downloader
  and dependency pinning.
- **#413** — the quant layout is resolved *and* validated against the on-disk byte counts
  (unknown layouts are refused rather than falling through to int2), shape-product overflow
  is rejected, and the olmoe dtype-3 path no longer trusts a crafted `nbytes` (heap overflow).

### Performance — all byte-identical

- **#481** +4.7x on the MLA-absorb score and value-mix reductions
- **#477** +13% decode on AVX-512 (`qt_addrow` / `qt_matvec_rows`)
- **#475** +11.6% with opt-in `XEXP=1` (one OpenMP region per expert block at S=1 full residency)
- **#473** +5.5% int4 IDOT at S=1 on AVX-512 VNNI

### Changed

- `glm.c` is now `colibri.c` plus header modules (#391); `make glm` remains as an alias.
- Serve stage 2 (#192): `response_format`, per-request grammars, grammar-forced drafts.

### Upgrade notes

- **`CTX` still defaults to 4096.** Coding clients send far more than that in a single system
  prompt. Use `CTX=32768`. Before this release an over-long prompt was silently truncated;
  now you get a clear 400 instead.

## [1.0.0] — 2026-07-19

First tagged release. The engine has been running in production since late June
2026; this tag marks the baseline for semantic versioning going forward.

### Highlights

- **GLM-5.2 (744B MoE)** runs on ~25 GB RAM in pure C, streaming experts from disk
- **Three-tier placement**: VRAM (hot) / RAM (warm) / NVMe (cold), with a learning
  cache that pins your workload's hottest experts automatically
- **CUDA backend**: multi-GPU expert tier, dense tensor distribution, batched
  ragged attention, resident pipeline (`COLI_CUDA_PIPE=2`)
- **Metal backend** (Apple Silicon): batched expert SwiGLU + fused decode attention
  on unified memory GPU
- **MTP speculation**: native GLM-5.2 draft heads, grammar-forced drafts, kernel-
  pinned verification (`SPEC_PIN=1`)
- **OpenAI-compatible API**: `coli serve` with SSE streaming, KV slots, bounded
  queue, web dashboard (`coli web`)
- **Web UI**: chat with live metrics, expert cortex brain page, profiling breakdown,
  expert atlas 3-D galaxy
- **Cross-platform**: Linux, macOS, Windows 11 (native MinGW), PowerPC; CI on all three
- **Auto-tune**: `coli plan --auto-tier` classifies the bottleneck and derives
  MTP/PIPE/NUMA/PIN settings with explanations

### Engine

- Token-exact validation against `transformers` oracle (teacher-forcing 32/32)
- Compressed MLA KV cache (576 floats/token, 57× smaller), persisted across
  restarts (`.coli_kv`, zero re-prefill)
- DSA sparse attention (lightning indexer), faithfully implemented
- Router-lookahead prefetch (`PILOT=1`, 71.6% predictive)
- Async expert I/O pool (`PIPE=1`), io_uring batching (`URING=1`)
- NUMA-aware expert placement (`COLI_NUMA=1`, +13–40% on multi-socket)
- AVX2 / AVX-512 / AVX-VNNI / ARM NEON / NEON-i8mm / POWER VSX kernels
- int4 / int8 / int2 / grouped-int4 (fmt=4) quantization formats

### Tools

- `coli convert` — FP8→int4 one-shard-at-a-time converter
- `coli doctor` — read-only setup diagnostics
- `coli plan` — resource planner with auto-tune prescription
- `coli bench` — MMLU / HellaSwag / ARC quality benchmarks
- Expert atlas (`tools/analyze.py --web`) — measured topic affinity for 19,456 experts

### Community

- 30+ hardware datapoints in the benchmark tracker
- Contributions from 20+ authors across engine, docs, tooling, and ports
