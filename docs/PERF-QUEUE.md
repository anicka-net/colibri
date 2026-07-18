# Performance queue — measured state and open work

State as of 2026-07-18, on a 2× H100 NVL + 2× 64-core EPYC host ("profile A" in
the hardware-tuning skill), GLM-5.2 int4, 64-token greedy decode, short prompt:

| Config | tok/s | Notes |
|---|---|---|
| `MTP=0`  | **19.6** | fused chain + grouped dual gemv + device accumulate |
| `DRAFT=1`| **17.9** @ 91% acceptance | S=2 verify forwards use the same fast paths |

Best-known env: `CUDA_DENSE=1 COLI_CUDA=1 COLI_GPUS=0,1 CUDA_EXPERT_GB=150
COLI_CUDA_PIPE=2 COLI_CUDA_PIPE_S_MIN=1 OMP_PLACES=cores OMP_PROC_BIND=spread`.

Measurement discipline: `TEMP=0`; expect ±3-5% run-to-run drift anyway (the
expert pin-placement stats file updates every run). `COLI_CUDA_PROFILE=1`
splits group time into H2D/kernel/D2H; `PROF=1` gives phase shares.

## Open items, largest first

### 1. DSA sparse attention + IndexShare (long context) — weights EXTRACTED, integration open
The snapshot has **no indexer weights** (`out-idx-*` never extracted), so every
layer runs full attention.  Native GLM-5.2 attends over the indexer's top-2048
(`index_topk`), refreshed every 4 steps, and `index_share_for_mtp_iteration=True`
reuses the selection for MTP drafts.  Below ~2k context this changes nothing;
past it, our attention grows linearly while native stays ~flat — and MTP verify
pays attention twice.
- Extraction: DONE 2026-07-18 — only 20 of 141 shards carry indexer tensors
  (~107 GB transient download, ~10 min on the fast host), output is 227 MB of
  out-idx files.  VALIDATED end-to-end: with `DSA_TOPK=32` forcing sparse
  selection at short context, greedy output is token-identical to full
  attention.  Caveat until integration: with the files present the engine
  auto-enables DSA and the current pipe/fused-chain gates cost ~3 tok/s at
  short context — set `DSA=0` on the discrete-GPU host for now.
  (Original plan, for reference: `--indexer` re-downloads shards
  to keep a few GB (resumable per shard).  Run it on the multi-GB/s host, not
  the 1 Gbps ones; the few-GB output then crosses the ~100 Mbps host-to-host
  link in minutes (never move the raw shards between machines).  The fast host
  has limited disk — fine for extraction (one ~5 GB shard in flight, deleted
  after conversion; output a few GB); anything bulk belongs on the RAID host.
- Integration: the CPU DSA paths exist, but the fused pipe2 chain has no index
  support and the pipe gate disables itself past `index_topk` when `has_dsa`.
- IndexShare for drafts comes nearly free once the above lands.

### 2. MTP multi-step drafts — RESOLVED, now an economics question
The old-tree "recursion bug" (23% at DRAFT=3) does not reproduce on current
main: DRAFT=2 gives 2.29 tok/forward (62% acceptance), DRAFT=3 gives 2.37
(44%) — inside the native 2.2-2.8 range.  Semantics were cross-checked against
the vLLM Glm4MoeMultiTokenPredictorLayer: same concat order, same recursion;
colibri's extra final_norm on h at step 0 measures BETTER than the
reference-exact variant (MTP_PRENORM=1: 53%/42%) — keep the default.
At short context deeper drafts lose on forward cost (S=3-4); the crossover
belongs to item 6 (long context), where halving forwards should win.

### 3. Fused-chain internal fusion
Attention is 1.5s/64 tok = 0.31 ms/layer, execution-bound across ~15 serial
small kernels.  Merging the small middle ops (kv-cache rmsnorm/copy/rope trio;
rope-Q into absorption) is the proven-shape next step.  CUDA graphs were
already tried and are NOT the answer (execution-bound, not launch-bound;
see COLI_GRAPH history).

### 4. Prefill profiling
All tuning so far targets decode.  The DS4/OpenAI server workload prefills
thousands of tokens (pipe1 `S>=8` path, never profiled here).  Time-to-first-
token may matter more than decode tok/s for agentic use.

### 5. Expert-group remaining headroom
Group kernel time is 761 ms/64 tok vs a ~100-150 ms bandwidth floor.  The
grouped dual gemv (templated RMAX 1/4) got it from 1533; vectorized weight
loads (uint4) or W4A16 tiles for rows 2-4 could close more.  Per-expert time
is now small enough that further work should be measured against item 3 first.

### 6. Long-context MTP crossover (in progress elsewhere)
At 70-token context MTP=0 wins (19.6 vs 17.9).  Attention grows with context,
drafts don't: find the crossover length; also where the device-resident KV
fix (O(context) upload removal) shows.

### 7. Spark-side cache policy
On the unified-memory host the oracle experiment bounds the prize: honest LRU
is 77% steady-state; Belady eviction alone 89-91%; oracle prefetch at ~89
loads/token → 100% within NVMe bandwidth.  A predictor needs ~65-90
experts/token a few tokens ahead; better eviction alone captures half.

## Measured dead ends (do not revisit without new evidence)
- CUDA graphs for the decode chain: execution-bound, graphs were slightly slower.
- `COLI_NUMA=1` weight interleave on decode: neutral (GPU-bound).
- NUMA-binding pinned staging: correct hygiene, neutral at decode volumes
  (`COLI_NUMA_STAGING=0` to disable; matters for batched/prefill).
- CPU-expert NUMA affinity: ~6% of expert calls are CPU, <1% ceiling.
- Fusing the shared expert into the chain: loses to overlap (17.0 vs 16.5) on
  strong-CPU hosts; knob `COLI_FUSE_SHARED=1` exists for weak-CPU machines.
