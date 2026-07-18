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

### 1. DSA sparse attention + IndexShare (long context) — short-context tax REMOVED, selection-in-chain open
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
  auto-enables DSA; since b9a1c30 the fused chain runs on indexer layers while
  selection is inactive and caches k_idx itself (DSA-on 19.7 vs DSA=0 20.2 —
  the residue is the CPU k_idx gemv, 19 'full' layers).  `DSA=0` no longer
  needed for short-context benchmarks.  REMAINING (the actual long-context
  win): selection inside the chain — device Ic cache, ix_wq/ix_wp scoring +
  top-k, absorption over an index list, 'shared'-layer selection reuse,
  index_topk_freq refresh — and lifting the pipe2 gate past index_topk.
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

### 4a. Cold-start (preload) anatomy — measured
Full cold start on the discrete host, 150 GB VRAM tier from tmpfs: **29 s**
(was 31 before parallelizing the tier upload): dense load 2.2 s, PIN placement
15 s (host slab warm-copy of ~213 GB — page-fault bound, ~5 CPU-min of sys
time — plus the now per-device-parallel VRAM upload), ~10 s misc (st_init,
CUDA init, wiring, KV).  On the unified host the same phase is NVMe-bound and
was the "absurdly slow start" complaint; the DS4 daemon (merged) amortizes it
— cold start is paid once per service start, not per request.  Remaining
levers if it matters again: zero-copy experts by mmapping the snapshot —
TMPFS ONLY (the pages are already resident RAM; the copy is pure waste and
skipping it is the actual point of tmpfs).  On NVMe hosts keep the eager
mlocked warm-copy: demand paging would trade a one-time load for page-fault
stalls in the decode path.  Also: THP/pre-fault for slab pages, pinned
double-buffer staging for the uploads.

### 4. Prefill profiling — MEASURED, now the top priority
2701-token prefill takes **246 s (11 tok/s)** — prefill runs at near-decode
speed; a 10k agentic prompt would be ~15 min.  Same with DSA=0, so it is not
the indexer.  PROFILED (2701 tok):
attention 114 s (score-softmax-value alone 93 s — the batch absorb kernel is
~60x off the FLOP floor and absorbed-MLA prefill is FLOP-heavier than
reconstructing k/v once; this is THE prefill fix — see design below) + expert-matmul 80 s + proj/rope 21 s + other 13.
FREE WIN: `COLI_CUDA_TC_W4A16=1` (lossless-weights tensor-core expert path,
rows>=16) cuts expert time 80 -> 40 s, prefill 207 -> 167 s (13 -> 16 tok/s);
greedy short-context output verified identical — recommended in the prefill
env until made default.  TC_INT4 (W4A4) is not worth it (200 s).  Long-context decode at 2.8k ctx:
DSA=0 **5.31 tok/s** (chain on, attention 115 ms/tok, linear in T);
DSA-on **2.45 tok/s** (selection active -> chain off -> CPU DSA path,
attention 44 s/128 tok).  Confirms item 1 phase 2 (selection in chain) is
mandatory for the model's purpose, and prefill needs work before that.
All tuning so far targets decode.  The DS4/OpenAI server workload prefills
thousands of tokens (pipe1 `S>=8` path, never profiled here).  Time-to-first-
token may matter more than decode tok/s for agentic use.

#### Prefill attention rewrite — design (ready to implement)
The 93 s is naive per-element contraction kernels; the same math as five
GEMMs is ~0.5 s of tensor-core time:
1. `qabs[S,H,kvl] = q_nope[S,H,192] @ kvb_k[h][192,kvl]` — int4-weight GEMM
   per head (w4a16_matmul shape);
2. `scores[h][S,T] = qabs[h] @ Lc^T + q_rope[h] @ Rc^T` — fp16 GEMM (new
   kernel; Lc/Rc tiles from the device KV cache, staged fp16);
3. causal online-softmax over scores (new kernel, row-tiled);
4. `ctxL[h][S,kvl] = P[h] @ Lc` — fp16 GEMM (same kernel as 2 transposed);
5. `out = ctxL @ kvb_v + o_proj` — int4-weight GEMM per head.
Steps 1/5 reuse the existing w4a16 wmma machinery; 2/4 need one fp16 GEMM
kernel; 3 is small.  Wire into attn_pipe_prefill behind an env
(COLI_PREFILL_GEMM=0 fallback), validate exact-vs-dense on a 2.7k prompt
(greedy continuation must match the current path), then default on.
Expected: prefill 167 s -> ~60-80 s (attention 114 -> ~20-30 incl. proj),
with the expert 40 s then the next target.

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

Direct CUDA execution from pageable LRU memory is now implemented behind
`COLI_CUDA_HOST_EXPERTS=1`.  On GB10, the 30 GB CUDA tier + cap 17 profile
improves the warm 32-token fixed benchmark from 38.53s to 25.12s and a
route-diverse 64-token run from ~80.6s to 52.3s.  Routed CPU expert time falls
from 23.79s to zero (CUDA critical path 2.07s), with byte-identical fixed
benchmark output.  The remaining dominant cost is felt expert I/O (9.8s/32
tokens, 21.5s/64 tokens), so cache prediction/eviction is the next Spark lever.
Removing the separate CUDA tier did not help: cap 36 holds fewer total experts
than the hybrid profile and produced the same ~24.8s warm latency.  `DRAFT=1`
adds a further ~9% on the direct 32-token run (1.61 -> 1.76 tok/s).

The published short-context Spark result is now reproduced on current main.
With `CTX=4096`, cap 63, no separate CUDA tier, host-backed experts and pipe2,
strict top-8 reaches **3.27 tok/s** over 208 tokens (89% hit).  `CACHE_ROUTE`
J2/M12 reaches **4.74 tok/s** (96.3% hit) by substituting 12.9% of route slots;
top-8 agreement is 87.1%, so keep this labelled as a quality-changing mode.
Safe `DRAFT=1` does not improve either condition: strict falls to 1.65 tok/s
because miss-containing groups must complete before their host slabs can be
reused, while CACHE_ROUTE remains 4.74 tok/s.  Host-wrapping the int8 MTP
experts removes the CPU fallback but does not change that verdict.

## Measured dead ends (do not revisit without new evidence)
- CUDA graphs for the decode chain: execution-bound, graphs were slightly slower.
- `COLI_NUMA=1` weight interleave on decode: neutral (GPU-bound).
- NUMA-binding pinned staging: correct hygiene, neutral at decode volumes
  (`COLI_NUMA_STAGING=0` to disable; matters for batched/prefill).
- CPU-expert NUMA affinity: ~6% of expert calls are CPU, <1% ceiling.
- Fusing the shared expert into the chain: loses to overlap (17.0 vs 16.5) on
  strong-CPU hosts; knob `COLI_FUSE_SHARED=1` exists for weak-CPU machines.
