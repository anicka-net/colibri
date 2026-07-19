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

### 1. DSA sparse attention + IndexShare (long context) — STRICT WIN, growing with T (+20% @6.7k, +63% @13.4k); open: IndexShare, decode-attention attribution (inc.6 ruled out selection cost)
The snapshot has **no indexer weights** (`out-idx-*` never extracted), so every
layer runs full attention.  Native GLM-5.2 attends over the indexer's top-2048
(`index_topk`), recomputed every token, and `index_share_for_mtp_iteration=True`
reuses the selection for MTP drafts.  (CORRECTION 2026-07-19: `index_topk_freq`
in the config is the FULL/SHARED *layer pattern* formula — freq=4/offset=3
generates exactly the 21-full `indexer_types` list; verified against the HF
reference, which has no temporal refresh.  Any decode-time refresh period is an
approximation, not native semantics.)  Below ~2k context this changes nothing;
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
2701-token prompt, discrete multi-GPU host, `COLI_PREFILL_GEMM=1 COLI_CUDA_TC_W4A16=1`,
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

Prediction accuracy still has a large theoretical prize.  At equal K6 budget,
offline exact-next-layer replay raises cap-17 demand hit rate 65.6% -> 84.5%,
while coupling reaches only 68.2%.  Live LOOKA telemetry explains the gap:
stale L+1 pilot recall is 69.7%, shared-expert two-step is 73.2%, and the
same-layer pre-attention predictor reaches 78.0%.  Timing dominates, however:
replacing the early pilot with same-layer K6 gives 1.33 tok/s, and combining
early K4 with late K2 gives 1.22 tok/s.  Both lose to early K6 at 1.37 tok/s.
Equal-budget K6 rank fusion also has no free win: mixing stale and two-step
predictions never beats pure two-step recall (60.6% on the measured decode),
while replacing two-step slots with previous-token routes degrades
monotonically (57.9% at one previous slot).  The predictors are redundant
where they are right; diversity from the weak temporal predictor is harmful.
Any next predictor must improve accuracy while retaining a full-layer I/O
horizon; late correction alone cannot hide the NVMe read.

### 7b. Precomputed prefix KV caches (agentic time-to-first-token)
Every new CLI/agent conversation re-prefills the same multi-thousand-token tool
system prompt.  At the measured prefill rates that is ~2 min for a 5k prompt —
paid per fresh session, and it dwarfs everything else in the interactive path.
The persistence format is ALREADY sufficient: `kv_disk_append` writes, per
token, the token id plus every layer's `Lc`/`Rc` **and the DSA `Ic` indexer
rows**, and `kv_hdr` binds layers/kv_lora/qk_rope/index_hd/n_ic/vocab, so a
restored prefix needs no re-scoring and a mismatched model is rejected.  Record
size is `4 + n_layers*(kv_lora+qk_rope)*4 + n_ic*index_hd*4` = **~215 KB/token**
here: 1.1 GB for 5k tokens, ~1 s to read from NVMe versus ~150 s to recompute —
two orders of magnitude.
What is missing is only the plumbing around it:
- **A keyed library** instead of one live file per slot: content-hash the
  leading token run, store `<cache>/<hash>.kv`, look up on request.  Today
  `$SNAP/.coli_kv` IS the conversation, and it is overwritten as the turn
  proceeds.
- **Read-only prefixes with copy-on-write**: loading a shared prefix must not
  let the continuing conversation append into it.
- **Longest-prefix match at submit time**, replacing the current
  compare-against-this-slot's-own-history (`mux_submit` / raw-API path).
- **Storage location**: on a tmpfs-snapshot host `$SNAP/.coli_kv` costs RAM;
  a prefix library belongs on real disk.
- Note RoPE bakes ABSOLUTE positions in, so a cache is only valid at position 0
  and two prefixes cannot be concatenated — fine for system prompts, which is
  exactly the use case.

### 8. Multi-user serving (reliability first, pipelining later)
Findings from the 2026-07-19 serve-path review.  What EXISTS: per-slot KV
isolation + prefix reuse in both serve modes (requests diff against slot
history, only the delta prefills — the `[API] KV slot N prefix a/b, prefill k`
line); frontend admission (capacity=KV_SLOTS, bounded queue, 429 overflow);
mux (`SERVE_BATCH=1`) batched ragged decode with fair per-token interleaving.
The rewind/shadow-staleness hazard (host truncation without clamping
`cuda_valid`/`cuda_ic_valid`) is FIXED via `kv_shadow_rewind` at all four
truncation sites.  Open items, in priority order:
- **Chunked prefill in mux**: a new submit's prefill runs synchronously in
  `mux_submit` and stalls every active stream — minutes at 100k+ context.
  Interleave decode iterations between prefill chunks.
- **Ragged-safe chain + DSA decode**: `step_decode_batch` passes `kvs[]`, and
  both the resident chain and the GPU selection decode are gated `!kvs` — so
  two simultaneous users drop to the slow ragged/CPU attention (and mux
  disables MTP).  Extending the chain to per-row KV states recovers
  single-user-class decode for small user counts.
- **Per-slot memory budget**: each slot costs host KV (~23 GB @131k) plus
  fp16 device shadows (~12 GB @131k) — KV_SLOTS × CTX collides with
  `CUDA_EXPERT_GB`; the adaptive-cap idea (borrow untouched reservations)
  generalizes here.
- **LONG-TERM (explicit goal, not today): pipeline users across GPUs /
  interconnected Sparks** — per-user home devices or layer-pipeline
  parallelism so concurrent users scale with hardware instead of sharing one
  decode batch.  Prereqs: the ragged-safe chain above, per-slot device
  residency, and (Spark) network-transparent expert tiers.  The planned Spark
  fabric is RoCE + NVMe-oF, which changes the shape of all three: the
  inter-stage payload is one residual vector (~24 KB/token — RDMA makes layer
  pipelining latency-trivial), a single NVMe-oF expert store serves every
  Spark (one snapshot copy; remote-miss latency makes prefetch lead time
  worth MORE than bandwidth), and a peer Spark's RAM over RDMA slots between
  local RAM and NVMe as a distributed expert-cache tier driven by the
  placement stats we already collect.

## Measured dead ends (do not revisit without new evidence)
- Spark 131k/cap-17 coupling prefetch: K1 is neutral; K2/K4/K8 regress as
  speculative reads evict useful experts.  Depth 2 also thrashes cap 63.
- Spark prefetch admission at the LRU end: predictions usually expire before
  demand; replay hit rate falls.
- Spark pilot budget below K6: K2/K4 save bytes but lose overlap and throughput.
- Spark late correction: same-layer K6 and early-K4+late-K2 arrive too late to
  hide NVMe latency.
- Spark predictor rank fusion: stale+two-step adds no recall at K6;
  previous-token+two-step is strictly worse.
- Spark strict MTP: extra routes plus miss-lifetime synchronization cut 3.27 to
  1.65 tok/s; under `CACHE_ROUTE` it is neutral.
- Spark cap 62 plus a 30 GB CUDA tier: both consume the same LPDDR and force
  swap.  Never combine the short-context cache with the production hot tier.
- Spark removing the CUDA tier while retaining the production-sized cache:
  cap 36 is neutral; the useful no-tier result requires the short-context
  cap-63 profile.
- Spark production W4A16 alone: pipe0 leaves almost all large-batch routed
  work on CPU, so the same frozen 260-token prefill stayed at 86 s.
- MTP-head route prediction: recall@8 is 6.2%; the final residual does not
  preserve early/mid-layer routing.
- CUDA graphs for the decode chain: execution-bound, graphs were slightly slower.
- `COLI_NUMA=1` weight interleave on decode: neutral (GPU-bound).
- NUMA-binding pinned staging: correct hygiene, neutral at decode volumes
  (`COLI_NUMA_STAGING=0` to disable; matters for batched/prefill).
- CPU-expert NUMA affinity: ~6% of expert calls are CPU, <1% ceiling.
- Fusing the shared expert into the chain: loses to overlap (17.0 vs 16.5) on
  strong-CPU hosts; knob `COLI_FUSE_SHARED=1` exists for weak-CPU machines.

## Next Spark experiments
1. **Adaptive expert-cache capacity — IMPLEMENTED, opt-in.**  The 131k planner
   reserves full-context KV and attention workspace up front, although their
   pages/cost grow with the live context.  `COLI_ADAPTIVE_CAP=1` borrows that
   untouched RAM at request boundaries and returns it before the request's
   declared maximum context can need it.  It is deliberately limited to
   `KV_SLOTS=1`, `COLI_CUDA_PIPE=0`, and non-mmap serve mode.  Truncated KV pages
   remain physically resident, so a monotonic high-water mark prevents unsafe
   cache regrowth; transient attention scratch can still be reclaimed.
   Offline trace replay: cap 17 = 66.2% LRU hit, short-request cap 42 = 87.6%.
   Live mux A/B with frozen placement, 131k production settings, 64 greedy
   tokens: **1.21 -> 1.43 tok/s**, hit 76.1% -> 83.7%, RSS 38.3 -> 67.1 GB,
   no swap.  Lifecycle probe grew 17->42, shrank 42->22 before a 100k-token
   request, then cancelled cleanly.  Deployed on the production mux service.
2. **Pipe0 CUDA prefill — IMPLEMENTED, opt-in.**  Production profiling first
   exposed that mux request timers started after `step()`, excluding the whole
   prefill; the corrected 2813-token run took 19m10s.  On a frozen 260-token
   prompt the pipe0 baseline was 86 s: expert matmul 56%, attention 29%, felt
   I/O 7%.  `COLI_CUDA_PREFILL=1` runs large contiguous attention and expert
   batches on CUDA without allocating pipe1/2's full-context device KV shadows.
   Splitting resident and missed expert groups lets async reads overlap the
   resident CUDA pass.  Result: **86 -> 64 s (26% faster)** with W4A16, or 67 s
   without it; the baseline and fast path produced the same 16-token greedy
   continuation.  The remaining cold-run bottleneck is expert I/O (26 s felt
   wait, 41% wall time).  The initial implementation kept DSA contexts beyond
   `index_topk` on CPU to avoid allocating the full-size `Ic` shadow.
   UPDATE after installing the missing 227 MB indexer artifact: the CPU DSA
   path took 15m38s on the frozen 2813-token prompt, versus 8m16s dense.
   Prompt-sized transient device `Ic` storage now lets the CUDA DSA prefill
   path run without a 131k shadow: **15m38s -> 5m53s (2.66x)**, also 29% faster
   than dense.  The first greedy token matched the CPU-DSA run; engine swap
   remained zero.  Attention fell from 74% to 21% of wall time, moving the
   bottleneck to CUDA expert matmul (54%).
3. **Learned low-rank early route correction.**  Train a small correction to
   the stale L+1 router logits from the same full-layer-horizon state.  Compare
   cross-prompt K6 recall against two-step and require enough gain to survive
   inference overhead before wiring any prefetch.
4. **Deadline-aware pilot queue.**  Instrument rank, enqueue layer, completion,
   demand use, and eviction.  If useful predictions are completing behind
   low-confidence work, prioritize by deadline/confidence and drop stale queue
   entries.  The K2/K4 regressions show that simply reducing breadth is wrong.
5. **Learned eviction/reuse scoring.**  Belady's 89-91% ceiling leaves real
   headroom over LRU's 77%.  Test offline using recency, frequency, layer, and
   routed-set context; only proceed if it transfers across prompts.
6. **Two-layer-horizon prediction.**  Evaluate exact and learned L+2 at equal
   K6 budget only after the above.  Coupling depth 2 already showed the failure
   mode: extra lead time is worthless when accuracy causes cache pollution.
7. **Blackwell sm_121 tensor-core formats (Spark prefill/batch only).**  Decode
   stays expert-streaming/hit-rate bound, so consumer-Blackwell features pay off
   only in the compute-bound paths — pipe0 prefill batches and the DSA TC-gather
   GEMMs.  Honest ranking:
   - **Block-scaled FP8 MMA (W4A8) — first.**  sm_120/121 add native MMA on
     block-scaled microscaling formats (mxfp8/nvfp4); our int4-with-block-scales
     is structurally the same thing.  Signed nibbles (−8…7) upcast EXACTLY into
     fp8 e4m3 (3 mantissa bits cover those integers), so an in-register
     int4→fp8 dequant feeding fp8 block-scaled MMA keeps weight fidelity while
     roughly doubling MMA throughput over the fp16 WMMA in `w4a16_*` /
     `gemm_f16_tc*` and halving activation traffic.  Only the activation
     quantization is lossy — per-block scales should keep it in the accepted
     fp16-class band, but verify with the frozen-usage greedy oracle.  The
     kernels to convert are exactly the Spark prefill hot spots (86→64 s pipe0
     result, 2.8k DSA prefill).
   - **nvfp4 — quality-gated experiment after.**  e2m1's value set
     {0, ±0.5…±6} is NOT a superset of int4 levels, so it needs requantization.
     This is "our W4A16 problem in hardware" — and the reason Hopper W4A4
     (`COLI_CUDA_TC_INT4`, measured dead end) lost: no block-scaled formats, so
     scales were applied in software and precision died.  Blackwell moves that
     into the MMA instruction.
   - **TMA bulk-copy pipelining — distant third.**  `dsa_gather_sel`, absorb,
     and prefill GEMM tiles stage global→smem by hand; TMA (Hopper-era, kept on
     sm_121) would improve compute/copy overlap, but the GB10 ceiling is shared
     LPDDR5x bandwidth, so expect modest gains.
   - **Offline ptxas for compute_121 — hygiene.**  Kills first-run JIT and gets
     Blackwell scheduling; low single digits at best.
   Caveats: sm_120/121 dropped thread-block clusters/wgmma-style paths, and the
   opt-in shared memory is ~99 KB vs H100's 227 KB — with the inc.5 smem
   formula ((2K+T+256)×4 B) the Spark dense-absorb ceiling is ~24k regardless
   of build arch, which makes DSA's topk-bounded smem MORE valuable on Spark
   than on Hopper.  sm_121 does NOT help: decode gemv (bandwidth-bound, int4
   already minimal bytes), expert-miss economics (NVMe/LPDDR bound), or the
   small-S MTP blocker (launch latency, not MMA format).

### Upstream weekend review (2026-07-19)

Imported two changes with direct value here:

- Upstream's measured-RSS guard (#403), hardened for this fork's asynchronous
  pilot and adaptive cache.  It uses current Linux RSS, drains queued/in-flight
  pilot loads before compacting LRU rows, frees CUDA host wrappers, and keeps an
  emergency ceiling so the next short request cannot regrow past a measured
  unsafe cap.
- The multi-seed DSA selection benchmark (#357), which removes deterministic
  quickselect pivot luck from randomized-shape speedups; the fixed plateau
  remains a single deterministic tie-stress input.  The local run completed
  with a stable 7-20x range rather than a single-input spike.

Deferred the resident ragged-KV patch: production uses `KV_SLOTS=1`, while
adaptive caching deliberately rejects multi-slot mode, so it cannot improve
the deployed path.  Revisit only with a measured concurrent/ragged workload.
Auto-NUMA and non-finite sampling already have local equivalents; converter,
release, Windows-prompt, and tool-calling changes do not affect this Spark
profile.

#### Increment 9 — serve-path hardening + attribution + the 5..15 gate (landed)
Four things from the 2026-07-19 serving/validation pass:
1. **32k VALIDATED** (26,844 tokens, CTX=32768, frozen usage): prefill
   1620.8 s — linear in token count vs 780 s at 13.4k — decode 2.09 tok/s,
   ZERO fallbacks.  First run past 2x the old cap.
2. **`kv_shadow_rewind`**: history truncation (serve prefix rewind, RESET,
   context-overflow reset) now clamps the device shadow watermarks
   (`cuda_valid`/`cuda_ic_valid`).  Before, a post-rewind prefill that hit a
   CPU-fallback layer left the shadow claiming validity over stale rows —
   silent corruption on the next decode.  Fixed at all four truncation sites.
3. **Turn-append smoke (the agentic number)**: serve protocol, 13.4k-token
   prefix, three turns.  Cold prefill 775 s; an 18-token append (over the
   rewind path) **7.8 s**.  Prefix/KV reuse across turns works end-to-end.
   The third turn (8-token append) exposed the **5..15 row hole**: the GPU
   selection paths were gated `S>=16` while the chain covers `S<=4`, so small
   appends — exactly the agentic pattern — ran CPU O(T)-per-row selection on
   all 78 layers: **80 s for 8 tokens**.  Gates relaxed to `S>4` (wmma tiles
   pad at 5..15 rows; still far ahead of CPU).
4. **Decode-attention attribution (the inc.6 puzzle, resolved)**:
   `COLI_DBG_DSACHAIN=2` adds CUDA-event phase timing to the chain.  At
   13.4k the chain's GPU time is ~120 ms/token — proj+kv+score 0.58 s,
   **absorb+o_proj 4.73 s (88%)**, tail 0.12 s over the smoke's decode — and
   the NON-chain residual is the three DENSE layers (L0-2), which bypass the
   chain, run per-token `attn_pipe_prefill` with an O(T) full absorb in the
   64-block launch shape, and inflate the misnamed projection/RoPE bucket
   (8.35 s @13.4k -> 16.56 s @26.8k, exactly linear in T).  NEXT decode
   levers, in order: (a) route the dense layers through the resident chain
   (kills the O(T)-with-sync-drain per token), (b) split-T/multi-block absorb
   launch shape (helps both dense absorb and the 1.4 ms/layer sel-absorb).

#### Increment 8 — device-side top-k for prefill selection (landed, default ON)
`COLI_DSA_DEVTOPK=1` (default; `=0` restores the host top-k) moves the
prefill selection top-k onto the device: `dsa_topk_rows` (one block per
query row) finds the exact keep-th-largest threshold with a 4-pass byte
radix select on a monotonic fp32 key, then emits indices with a stable
block-scan compaction — strictly-greater in position order, ties in
position order until keep — i.e. the host `partial_select` semantics
BIT-EXACTLY.  Scoring is chunked (512 rows), so the S_b×T score matrix
never exists anywhere: not as the device scratch (which stops allocating
past ~64k), not as the host staging buffer (611 MB at 13.4k, 67 GB at
131k — the silent CPU-fallback trigger), and not as the S_b×T×21-layer
PCIe download (260 GB at 256k).  Only the selection itself moves
(S_b×topk ints, independent of T).  nsel is host-computed (min(nk,topk)).
Scratch slot 40 (table widened to 44); no new exports.
MEASURED at 13.4k (frozen usage): greedy output TOKEN-IDENTICAL to the
host path (same device scores -> same selection), prefill 780.1 vs
780.7 s (parity — the downloads were already overlapped at this T), RSS
−0.6 GB (the host score buffer), fb 0.  The payoff is that 64k-256k
prefill selection now runs on GPU at all.
Follow-up (deferred): keep the selection device-resident across the
FULL->SHARED layers of one batch to skip the per-layer re-upload
(needs per-device stamping + a lazy host download for mixed CPU
fallbacks; the upload is topk-bounded, so it can wait).

#### Increment 7 — fp16 KV/Ic device shadows (landed, default ON)
`COLI_KV_F16=1` (default; `=0` restores fp32) stores the device KV shadows
(`cuda_Lc`/`cuda_Rc`) and the DSA indexer shadow (`cuda_Ic`) as `__half`:
half the shadow VRAM and half the read bandwidth in every consumer.  This is
the 256k enabler — fp32 shadows would cost ~48 GB total at 256k (a third of
the expert budget); fp16 makes it ~24 GB.  The HOST canonical caches stay
EXACT fp32: chain and prefill writers compute rows in fp32 staging (scratch
slots 37-39), download the exact fp32 to host, and convert only the device
copy; uploads go through `coli_cuda_pipe_upload_kv`/`copy2d_kv` (new
exports, Windows loader entries done).  Readers are templated on the storage
type — absorption/sel/batch kernels, `dsa_score`, `dsa_gather_sel`, and the
`gemm_f16_tc` family, which now feeds the tensor cores DIRECTLY from fp16
(the fp32→fp16 smem staging conversion disappears; `LcSel`/`RcSel` gather
buffers are fp16 too).  Host-KV staging paths (`absorb_batch`,
`project_batch`, ragged) stay fp32; `pfg_test` pins `COLI_KV_F16=0`.
MEASURED at 13.4k (frozen usage, 64 greedy tokens, fb 0 both modes):
decode 2.95 (f32) vs 2.91 (f16) tok/s and prefill 781.6 vs 778.0 s — parity
within drift; prefill score-softmax-value 87.7 → 84.5 s; VRAM −0.6 GB/GPU
(the whole shadow is only ~1.2 GB/GPU at this T — the win is capacity at
32k+).  Numerics: same first-decode winner and top-8 set, logit shifts
≤0.6 = the accepted fp16 class (as W4A16/TC); greedy text flips a near-tie
on the degenerate repetitive longprompt, as every fp16-class change does.
Follow-up idea: int8 shadows would halve again but need per-row scales in
every reader — only worth it if 256k VRAM pressure demands it.

#### DSA phase 2, increment 6 — COLI_DSA_REFRESH knob + selection-cost attribution (landed, default off)
Motivated by the (now corrected, see item 1) belief that native semantics
refresh selection every 4 steps.  `COLI_DSA_REFRESH=N` (default 1 = native
per-token selection, zero change): FULL layers recompute top-k every Nth
decode token; between refreshes they reuse the layer's cached selection
extended with the new tail positions (recent tokens are never dropped), with
per-layer caches, rewind/slot invalidation, and k_idx still appended to the
Ic shadow every token.  Chain gains a `score_off` mode (k_idx append without
scoring/sync/top-k); new exported accessor `coli_cuda_dsac_times` went
through the 4-entry Windows loader ritual.
MEASURED at 13.4k (frozen usage, 64 greedy tokens): REFRESH=4 reuses 846 of
1134 FULL engagements, decode **2.94 -> 2.98 tok/s (~1%, within drift)** —
and the greedy continuation DIVERGES.  Verdict: quality-affecting for a
noise-level gain at this T — near-dead-end, keep default 1.  The knob's real
product is the attribution (`COLI_DBG_DSACHAIN=1` now prints cumulative
mid-sync and host-top-k time): sync **0.54 s** + top-k **0.13 s** of 21.5 s
decode = **3%**.  The selection machinery is NOT the DSA decode T-growth
term at 13.4k, which also de-prioritizes device-side top-k for DECODE
(prefill still needs it: the S_b×T score download grows ~20x by 256k, and
host top-k is O(T) per token so revisit past ~64k).  Where decode attention
(16.4 s/64 tok) actually goes, per the profile: projection/RoPE 8.3 s +
~7.8 s absorb-side — both nominally T-independent, yet short-context
attention is 1.5 s/64 tok.  NEXT PROBE: attribute the chain's per-phase
time at long T (suspects: `attention_absorb_sel_kernel`'s H=64-block launch
shape serializing 2048 rows per block, and what the projection timer
actually covers in the chain path).

#### DSA phase 2, increment 5 — T>8192 cap lift (landed)
The absorb-batch kernel family's `T<=8192` checks were shared-memory limits
in disguise ((2K+T+256) floats of dynamic smem vs the 48 KB default), not
algorithmic ones: `absorb_smem_ok()` now computes the actual need and raises
the kernel's dynamic-smem attribute to the device opt-in maximum when needed
(227 KB on Hopper -> T up to ~56k for dense absorb; past that the functions
refuse and the CPU fallback engages as before).  The memory-bound paths
(`coli_cuda_prefill_attn_gemm`, the `attn_pipe_prefill` gate) lift straight
to T<=131072; the DSA decode chain never had a T cap (sel-absorb is over
topk); KV/Ic device shadows already size by CTX (~1.4 GB/GPU at 16k).
MEASURED at 13.4k ctx (2x the 6.7k prompt, CTX=16384, frozen 2.7k-warmed
usage, 64 tokens): DSA prefill **780.0 s** vs dense 871.9 (-10.5%); decode
DSA **2.98 tok/s** vs dense 1.83 (**+63%**, was +20% at 6.7k — the flat-vs-
linear divergence the cap was hiding), attention 16.4 vs 29.8 s/64 tok,
fallbacks 0, prefill engagement 21/57.  Dense decode at 13.4k itself now
runs on GPU via the smem opt-in (pre-lift: CPU path).  Both configs pay
~65-69% expert hit (usage frozen on the short prompt), so cross-config
ratios are the trustworthy numbers.  (Increment 6 then measured the
selection machinery at 3% of decode — the growth lives elsewhere; see
its entry above.)

#### DSA phase 2, increment 4 — prefill sel-absorb TC gather (landed)
Phase-B rows now gather their selected Lc/Rc rows into contiguous buffers
(`dsa_gather_sel`, one block per (row,key), amortized across all 64 heads —
the scalar kernel re-read the scattered latent per head) and absorb via
z-batched fp16 TC GEMMs with heads as the M dimension: per chunk of 32 rows,
scores[H,topk] = qabs @ LcSel^T + q_rope @ RcSel^T (`gemm_f16_tc_zb`), flat
softmax, ctxL = P @ LcSel, with the per-head int4 projections as
block-diagonal wmma variants (`w4a16_nn_scaled_bd`/`w4a16_nt_bd`,
blockIdx.z=head).  Scratch slots 32-36 (table widened to 40); no new
exports, so no Windows loader entries.  `COLI_DSA_TCGATHER=0` keeps the
scalar kernel.  MEASURED: 2.7k prefill 70.1 -> **51.7 s** (attention 29.6 ->
11.0); 6.7k prefill 523.2 -> **392.2 s** vs dense 411.1 — DSA at 6.7k is now
a strict win (prefill -4.6%, decode +20%, fb 0), attention core 164.5 ->
32.8 s (5x).  VALIDATION: pfg_test grew a sel-path A/B (scalar sel-absorb as
reference, same inputs, same lists): worst per-row rel err 2.8e-3 @topk=128
/ 2.6e-3 @topk=2048 (mag 1x), 6.3e-2 @mag 30x — the same fp16-rounding class
as the landed prefill GEMM.  In-model 16-token greedy continuation
bit-identical to the scalar path at 2.7k.
WAR STORY (cost: one wedged H100 + a host reboot): the first harness
version under-allocated the sel buffer — `coli_cuda_prefill_attn_gemm`
reads `sel_host+sB0*topk`, i.e. the array covers ALL S rows like
`m->dsa_sel`, not just phase-B rows.  The host OOB shipped garbage indices
to the GPU; wild latent reads wedged the kernel unrecoverably (unkillable
R-state process, nvidia-smi hung, no watchdog on datacenter GPUs).
`dsa_gather_sel` now clamps indices to [0,T) — a corrupt selection can
produce wrong output but never touch wild VA.  Post-reboot, CUDA refused to
init (error 3, zero diagnostics anywhere) until `nvidia_uvm` was reloaded
with `uvm_disable_hmm=1` — the host's modprobe.d had that option for a
reason (open-gpu-kernel-modules #780/#797).

#### DSA phase 2 — 6.7k benchmark + DRAFT=1 composition (measured 2026-07-19)
Four runs on the discrete multi-GPU host, 6711-token prompt (2701 for the short DRAFT point),
frozen usage warmed on the 2.7k prompt, TEMP=0, NGEN=128:

| config | prefill | decode | hit | notes |
|---|---|---|---|---|
| dense @6.7k | 411.1 s | 3.19 tok/s | 71.9% | attn 31.1 s/128 tok |
| DSA @6.7k | 523.2 s | **3.82 tok/s** | 73.6% | attn 24.7 s; fb 0 |
| DSA+DRAFT=1 @2.8k | 75.4 s | 4.85 tok/s | 90.6% | 88% acc, 1.88 tok/fwd |
| DSA+DRAFT=1 @6.7k | 547.5 s | 4.02 tok/s | 69.9% | 84% acc, 1.86 tok/fwd |

Findings: (1) **DSA decode advantage is real and grows with T**: +20% at
6.7k (was −13% at 2.8k), attention 24.7 vs 31.1 s — the crossover is behind
us and DSA stays ~flat while dense grows linearly.  Engagement clean, zero
fallbacks in all runs.  Both configs pay the same ~72% hit-rate penalty
(usage frozen on 2.7k), so the ratio is trustworthy.  (2) **DSA prefill
REGRESSES at 6.7k**: +112 s vs dense, all of it attention (169.2 vs 58.4 s,
score-softmax-value 164.5 vs 53.9 s).  Cause: phase-B rows use the batched
sel-absorb kernel (plain fp loads over ns=2048/row) while dense rides the
TC GEMM; at 6.7k mean causal length ~3.3k is only 1.6x the sel width, so
the op saving cannot cover the per-op throughput gap.  Estimated crossover
without a fix ~20k+.  Fix direction: gather the selected KV rows into a
contiguous buffer and run the absorb as a TC GEMM per row-tile (or extend
W4A16 TC path to the gathered case).  (3) **MTP composes with DSA
correctly** — 88%/84% acceptance, selection reused across draft rows, fb 0
— but end-to-end gain is only ~4-5% because an S=2 forward costs nearly
2x an S=1 forward (the known small-S GPU-forward blocker; MTP payoff
remains parked on it).  Decision from the numbers: prefill sel-absorb TC
gather is now the top DSA lever (blocks recommending DSA-on as default);
the T>8192 cap lift is what exploits the decode win at 16k+;
index_topk_freq=4 demoted (decode already ahead of dense).

#### DSA phase 2, increment 3 — per-row PREFILL selection (landed)
`attn_pipe_prefill` now handles batches crossing/past index_topk with a phase
split: rows with nk<=topk keep the five-GEMM causal path (S_a rows, T_a keys),
rows past it get per-row selection and a batched sel-absorb
(`attention_absorb_sel_rows_kernel`, one block-row per query, fixed ns=topk —
an invariant of the top-k).  FULL layers run the indexer on-device
(`coli_cuda_prefill_dsa_select`: batched k_idx into the Ic shadow + qi/w32 +
`dsa_score_kernel` over all rows, one score download ~7 MB @2.7k) with the
exact host top-k (`prefill_dsa_topk`, OMP over rows); SHARED layers reuse
`m->dsa_sel`.  Dense layers L0-2 ride the same path through the
attention_rows hook (scratch out + download).  The pipe2 and hook gates are
lifted for S>=16; 5..15-row batches and any missing prerequisite fall back to
the whole-layer CPU path — never dense attention under selection.
MEASURED (2701-token prompt, frozen usage): **prefill 267.8 -> 70.3 s
(38 tok/s, 3.8x)**; decode unchanged-to-better at 4.79 tok/s; engagement
prefill full 21/21 shared 57/57, fb 0.  vs DSA=0: prefill 54 s, decode 5.50 —
DSA-on is now within ~30%/13% at the 2.8k crossover and flat in T beyond it.
VALIDATION: text identity is not a usable bar here — top-32 of a 91-token
prompt is genuinely lossy (even CPU-DSA != dense), and fp tie-flips diverge
text between scorers.  Instead `COLI_DBG_SELDUMP` records per-row selection
lists; GPU-vs-CPU runs at DSA_TOPK=32: layer-0 sets (bit-identical inputs)
are **100% identical**; overall mean overlap 96.5%, degrading smoothly with
depth = accumulated residual fp drift (same accepted class as
COLI_PREFILL_GEMM), while a structural bug would read ~35% (random).
Remaining prefill delta vs dense (~16 s): phase-B absorb gather + scoring +
per-FULL-layer score sync.  Next DSA levers: index_topk_freq=4 refresh
(decode), IndexShare for MTP.

#### DSA phase 2, increment 2 — selection inside the resident chain (landed)
The pipe2 gate is lifted for decode batches (S<=4) entirely past index_topk:
`coli_cuda_pipe_attn_chain` gained a DSA mode (`ColiCudaDsaChain`).  FULL
indexer layers compute the new k_idx row (LayerNorm+RoPE into a per-layer
device Ic shadow, `ic_dev_sync` mirrors `kv_dev_sync`) and score all context
tokens on-device (`dsa_score_kernel`), then ONE mid-chain sync downloads the
score row (~11 KB @2.8k) for the exact host top-k (`chain_dsa_topk` — same
partial-select + position-order tie-break as the CPU path) and uploads the
sel list; absorption runs the inc.1 sel-absorb kernel with device-resident q.
SHARED layers upload the last FULL layer's sel and absorb over it.  Dense
layers L0-2 (all FULL) keep CPU selection each step, which is what feeds the
first shared chain layers.  Fallback on any missing prerequisite is the whole
layer on CPU (full semantics), never dense-under-selection.
`COLI_DSA_CHAIN=0` restores the old gate; `COLI_DBG_DSACHAIN=1` prints
engagement counters (full/shared/fallback + reason).
GOTCHA found on the way: the indexer extraction stores ix_* weights as INT8
(fmt 1, per-row scales), not int4 — the chain's indexer projections use the
generic `quant_matmul` (fmt 1/2); a hard `fmt==2` check made every FULL layer
silently fall back (engagement counters caught it; the run was correct but
slow, 2.51 tok/s, because CPU fallback preserves semantics).
MEASURED (2701-token prompt, frozen warmed usage, TEMP=0, MTP=0):
decode @2.8k **2.48 -> 4.67 tok/s** (2.0x over the 2.34 pre-phase-2 state);
attention 42.8 -> 19.5 s/128 tok; aproj 26.5 -> 3.6 s (the 3 dense layers'
CPU selection), score-softmax-value 15.2 -> 0.6 s, chain itself ~15.3 s/128
(~1.6 ms/layer incl. the per-FULL-layer sync + host top-k).  Engagement
verified: full 18/18, shared 57/57, fallback 0.  Text identical to dense and
to the CPU-selection path at DSA_TOPK=32.
Dense attention at 2.8k is still 5.50 tok/s, so DSA-on decode is now within
15% of dense at the crossover context and stays ~flat in T while dense grows
linearly.  Remaining levers (inc.3 candidates): index_topk_freq=4 refresh
(amortize the 18 per-token sync+topk to every 4th step — native semantics),
chain-side selection for the 3 dense layers, DSA-on PREFILL (still 268 s,
non-pipe2 absorb path — per-row prefill selection is the big one), and
IndexShare for MTP drafts.

#### DSA phase 2, increment 1 — GPU absorb+o_proj over the selection (landed)
`coli_cuda_attention_project_sel` (absorb over the indexer's top-k list +
fused o_proj, device KV shadow, selection still CPU/bit-identical) now
serves decode rows where every row has a selection.  Validated: DSA_FORCE
top-32 short-context text identical to dense through the new path.
MEASURED at 2.8k: decode 2.34 -> 2.48 tok/s only — the pie shifted, not
shrank: attention/128tok = 26.5 s CPU (projections + k_idx + selection
scoring + top-k) + 15.2 s GPU (78 sequential upload->kernel->sync round
trips, latency-bound; CPU o_proj timer now 0).  CONCLUSION for the next
increment: per-stage offload cannot beat the latency wall — the win is
keeping the PIPE2 CHAIN resident past index_topk with (a) selection
scoring in-chain on the device Ic shadow, (b) CPU top-k on a downloaded
score row (32 KB, keeps bit-identical selection), (c) the new sel-absorb
kernel called with device-resident q (no per-layer host round trip),
(d) the pipe2 gate lifted.  The kernel/entry/loader plumbing from this
increment is the building block; est. decode at 2.8k -> ~5.5+ tok/s flat
in T (selection caps attended keys at 2048).
