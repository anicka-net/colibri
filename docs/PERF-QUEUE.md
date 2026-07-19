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

#### Prefill attention rewrite — IMPLEMENTED (COLI_PREFILL_GEMM=1)
The five-GEMM tensor-core rewrite of score-softmax-value is in
(`coli_cuda_prefill_attn_gemm`, backend_cuda.cu): per head, (1) qabs =
q_nope @ kvb_k (int4-weight wmma, per-reduction-row scales), (2) scores =
qabs @ Lc^T + q_rope @ Rc^T (fp16-TC GEMM, fp32 accumulate), (3) causal
softmax (tail zeroed), (4) ctxL = P @ Lc, (5) ctx slice = ctxL @ kvb_v,
then one o_proj GEMM over the assembled context.  Hooked into
attn_pipe_prefill for `out_dev && S>=16` (decode/MTP keep the absorb
kernel); `COLI_PREFILL_GEMM=0` falls back, `=2` launches on the legacy
stream (debug).  MEASURED on the 2701-token prompt:
**prefill 161.7 -> 80.4 s (16.7 -> 33.6 tok/s), score-softmax-value
93.0 -> 11.9 s**, attention 114 -> 33 s.  Numerics validated against the
absorb kernel in a standalone harness (S/T up to 2701, model dims,
S<T windows, 30x magnitudes: worst-row rel err ~1e-3..7e-2 = fp16
input rounding, no structure).  Token-exact greedy match vs the old path
is NOT a usable bar: expert placement (`.coli_usage` evolves per run)
makes even two identical GEMM=0 runs diverge after ~16-20 tokens.
Remaining prefill costs: proj/rope 21 s (pipe_gemm gemv-style — same
w4a16 treatment applies), experts 40 s, other 13 s.
UPDATE: pipe_gemm now routes int4 tensors with S>=16 through the w4a16
wmma tiles (`COLI_PIPE_TC=0` opts out): projection/RoPE 21.0 -> 1.7 s,
**prefill 78.4 -> 54.0 s (50 tok/s)** on the 2.7k prompt — the shared
expert's prefill gemms ride the same path.  Decode (S<16) stays on the
exact fp32 kernel.  Perturbation vs fp32 projections: ~0.06 on the top-1
logit at 2.7k, same winner; frozen-state pairs are bit-identical.

#### Measured dead end: next-token expert prediction from the MTP head
`COLI_DBG_MTPROUTE=1` (telemetry, DRAFT>=1) runs every layer's router on
the MTP block's predicted hidden state and scores it against the next
forward's true routing.  Result at 2.7k ctx over 256 tokens: recall@8
6.2%, recall@16 10.2% (random = 3.1%); mid-stack layers ~0%, late layers
15-20%.  The MTP state approximates the FINAL residual and the residual
direction rotates too much through the stack to drive early/mid routers.
Cross-token same-layer prediction (the NLA work) is the viable route for
the NVMe prefetcher; keep this probe around to re-measure variants.

#### FIXED: softmax shared-memory race = the run-to-run nondeterminism
Root cause found and fixed (2026-07-19).  Three CUDA attention kernels
(`attention_absorb_batch_kernel`, `attention_absorb_ragged_kernel`, and
the fused-chain `absorption_kernel`) read the block max (`mx=red[0]` /
`max_s=warp_vals[0]`) and then re-used the same shared slot for the sum
reduction WITHOUT a barrier in between: a fast warp could store its
partial sum over the max before a slow warp had read it, silently
mis-scaling part of that (head,row) block's softmax.  Effects observed
before the fix: a handful of residual rows perturbed at ~1e-5..1e-3 per
forward (first visible at the dense layers), amplified by routing
discreteness across 78 layers into ±0.5 logit jitter at 2.7k context —
enough to flip greedy near-ties.  On a degenerate repetitive prompt the
flip cascades ("</think></think>}}}..."), and the degenerate run's
routing stats then poison `.coli_usage`, dropping the next runs' pin hit
rate (observed 92% -> 63%) and amplifying variance further.  The earlier
"contiguous tail corruption from row ~272" reading of TAP dumps was an
artifact: rows past the prompt length are DECODE steps, which diverge
wholesale after any first-token flip.  The Metal shader already had the
barrier; only CUDA was affected.  Fix: one `__syncthreads()` per kernel
between the max read and the sum store.  With the fix + a frozen
`.coli_usage`, repeated greedy runs are bit-identical (logits and text).
Benchmarking notes from the hunt:
- `.coli_usage` evolves between runs by design — freeze/restore it for
  cross-run comparisons.
- The per-run "expert hit rate" scores the CURRENT run's routing against
  the pinned set: a degenerate decode routes to unusual experts and
  REPORTS a low hit rate (effect), but a sustained batch of degenerate
  greedy runs also WRITES enough skewed selections to genuinely damage
  the learned placement (~30 long benchmark runs took the discrete host
  from 92% to ~65% on long prompts) — both directions are real.  A
  fresh file costs ~25 points until history relearns; normal TEMP=0.7
  service traffic rebuilds it organically.
- Greedy (TEMP=0) raw continuation of repetitive or list-like text
  (the 10x-repeated-paragraph test prompt, doc files) degenerates into
  loops as a MODEL behavior — with the race fixed this is deterministic,
  not flaky.  Long-context quality checks should use the chat template
  and default temperature; keep TEMP=0 for numeric A/B only.

### 5. Expert-group remaining headroom
Group kernel time is 761 ms/64 tok vs a ~100-150 ms bandwidth floor.  The
grouped dual gemv (templated RMAX 1/4) got it from 1533; vectorized weight
loads (uint4) or W4A16 tiles for rows 2-4 could close more.  Per-expert time
is now small enough that further work should be measured against item 3 first.

### 6. Long-context MTP crossover (in progress elsewhere)
At 70-token context MTP=0 wins (19.6 vs 17.9).  Attention grows with context,
drafts don't: find the crossover length; also where the device-resident KV
fix (O(context) upload removal) shows.
DATA POINT (2026-07-19, post-determinism-fix, frozen warmed usage,
2701-token prompt): DRAFT=1 **5.57 tok/s** (90% acceptance, 1.91
tok/forward) vs MTP=0 **5.50 tok/s** — MTP is already (barely) ahead at
2.8k, so the crossover sits below that; the gap should widen with T.

#### Long-context rerun after the determinism fix (canonical numbers)
2701-token prompt, tekton, `COLI_PREFILL_GEMM=1 COLI_CUDA_TC_W4A16=1`,
frozen usage warmed on the workload (hit 82-93%):
| config | prefill | decode @2.8k |
|---|---|---|
| MTP=0 DSA=0 | 78.4 s | **5.50 tok/s** |
| MTP=0 DSA=0 (repeat, same state) | 79.2 s | 5.47 tok/s, TEXT IDENTICAL |
| DSA-on (selection active) | 268.5 s | 2.34 tok/s |
| DRAFT=1 DSA=0 | 83.5 s | **5.57 tok/s** @ 90% acc |
The identical-text repeat is the determinism guarantee in action; ±1%
timing drift remains (thermal/scheduling).  DSA-on is still gated OFF
the pipe2/GEMM paths past index_topk (prefill 268 s = non-pipe2 absorb
path) — item 1 phase 2 (selection in chain) stays the top long-context
lever, now worth ~2.4x decode AND ~3.4x prefill at 2.7k.

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

Coupling prefetch transfers across workloads.  A table trained on the first
70% of a prime-number trace predicts an unrelated compiler prompt; its single
top prediction has 83% precision.  On Spark, the existing
`PILOT_REAL=1 COUPLE_K=8 COUPLE_D=1` default improves strict decode from 3.17
to **3.58 tok/s**, raises hit rate 87.8% -> 92.3%, and cuts felt I/O wait
20.9 -> 15.0 s over 128 tokens.  K=1 is slightly slower at 3.51 tok/s despite
fetching fewer bytes (147 vs 172 GB), because it leaves less I/O overlapped.
Depth 2 over-prefetches and thrashes: K8/D2 falls to 2.11 tok/s, 79.5% hit,
and 458 GB fetched.  Keep `COUPLE_D=1`.

This gain does not transfer to the 131k production profile with a 30 GB hot
tier and cap 17.  There, no coupling and K1/D1 both measure 1.35 tok/s
(18.1/18.0 s felt I/O wait), while K8/D1 regresses to 1.25 tok/s and fetches
442 GB instead of 329 GB over 64 tokens.  Keep coupling disabled in that
service; it currently pays only when the short-context profile can devote
roughly 92 GB to its LRU.

Admission/budget sweep after the determinism fix confirms the boundary.  On
the 131k cap-17 profile, coupling K2/K4/K8 gives 1.34/1.32/1.25 tok/s versus
the 1.37 baseline, with fetched bytes rising monotonically.  Inserting
prefetches at the LRU end also loses in offline replay, because useful
predictions are evicted before demand reaches them.  The existing stale-state
pilot budget is already at its measured knee: PILOT_K 2/4/6 gives
1.21/1.29/1.37 tok/s.  Keep `PILOT_K=6` and coupling off for this profile.

## Measured dead ends (do not revisit without new evidence)
- CUDA graphs for the decode chain: execution-bound, graphs were slightly slower.
- `COLI_NUMA=1` weight interleave on decode: neutral (GPU-bound).
- NUMA-binding pinned staging: correct hygiene, neutral at decode volumes
  (`COLI_NUMA_STAGING=0` to disable; matters for batched/prefill).
- CPU-expert NUMA affinity: ~6% of expert calls are CPU, <1% ceiling.
- Fusing the shared expert into the chain: loses to overlap (17.0 vs 16.5) on
  strong-CPU hosts; knob `COLI_FUSE_SHARED=1` exists for weak-CPU machines.
