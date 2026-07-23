# Fix log

## Round 1 - Vojtech workspace reconciliation

### 1. INT4-only fused decode path

**NOT DONE** - The attempted duplicate prologue was removed. Current `main`
already provides the format-aware PIPE2 resident path, and the imported path
was unreachable whenever PIPE2 succeeded.

### 2. Per-slot CUDA KV shadows

**FIXED** - CUDA shadow pointers and validity now belong to each `KVState`.
Ragged batches bypass this optimization, and slot switches retain independent
device histories.

Real-model two-slot evidence:

```text
0 alpha {'prompt_tokens': 13, 'completion_tokens': 1, 'total_tokens': 14}
1 beta {'prompt_tokens': 13, 'completion_tokens': 1, 'total_tokens': 14}
0 alpha {'prompt_tokens': 13, 'completion_tokens': 1, 'total_tokens': 14}
OpenAI-compatible API listening on http://127.0.0.1:18000/v1
[API] KV slot 0 prefix 8/13 token, prefill 5
[API] KV slot 1 prefix 8/13 token, prefill 5
[API] KV slot 0 prefix 13/13 token, prefill 0
```

### 3. Unsafe asynchronous argument ring

**NOT DONE** - The old global staging ring was not imported. The fork retains
its existing synchronized per-device expert batching.

### 4. Optional Windows DLL symbols

**NOT DONE** - The managed-memory and duplicate decode-prologue exports were
removed with the superseded port, so the fork's DLL ABI did not change.

### 5. Partial CUDA KV allocation

**CHANGED-UNVERIFIED** - Pair allocation now publishes pointers only after
both allocations succeed and frees either temporary on failure. There is no
existing CUDA allocation fault injector to force the half-success path.

### 6. INT4 server launch arguments

**FIXED** - `openai_server.py`, `coli serve`, and `coli web` accept
`--expert-bits` and `--dense-bits` (plus environment equivalents) and pass
both values to `glm`.

Focused test evidence:

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
```

Real-model `coli serve --expert-bits 4 --dense-bits 4` evidence:

```text
alpha {'prompt_tokens': 13, 'completion_tokens': 1, 'total_tokens': 14}
OpenAI-compatible API listening on http://127.0.0.1:18000/v1
```

### 7. Per-slot CUDA cleanup

**FIXED** - `serve_ctx_free()` releases each slot's device shadows on the
layer's owning device before freeing the pointer arrays.

Graceful legacy-server shutdown evidence:

```text
[API] KV slot 0 prefix 5/10 token, prefill 5
response alpha {'produced': 1, 'tokps': 0.1, 'prompt_tokens': 10, 'truncated': 0}
exit 0
```

### Full validation

CPU:

```text
OK kv_alloc re-allocation
----------------------------------------------------------------------
Ran 72 tests in 2.078s

OK
```

CUDA on NVIDIA GB10:

```text
[CUDA] device 0: NVIDIA GB10, 130.7 GB VRAM, sm_121
cuda backend: q8/q4/q2/f32 correctness ok on 1 device(s)
```

### Self-check

1. Item 5 was not failure-injected because the backend has no allocation
   fault-injection hook. All other retained behavior, including slot cleanup,
   was run.
2. Every FIXED or CHANGED-UNVERIFIED claim corresponds to the final diff and
   the command output above.
3. The review required CPU build/tests, CUDA compilation/runtime, format-safe
   integration, KV-slot isolation, and staging lifetime safety. CPU, CUDA, and
   slot outputs are above; the two superseded CUDA features were not imported.

## Round 2 - dev to main reconciliation

### 1. Greedy NaN argmax

**FIXED** - Greedy and MTP selection now share a finite-only argmax. The
regression test injects `NaN` at token 0 and verifies that token 123 wins.

```text
greedy + MTP: non-finite logits skipped, finite argmax wins ok
test_sample_nan: ok
```

### 2. `coli stop` process matching

**FIXED** - Process discovery parses NUL-separated argv, requires an actual
`coli serve` argv pair, and limits engine fallback matching to
`COLI_MODE=serve` with the requested `SNAP`. Pidfiles now live in a private
runtime directory with mode `0600`.

```text
test_matches_only_real_coli_serve_argv ... ok
test_pidfile_is_private_and_outside_shared_tmp ... ok
```

### 3. Watchdog hangs and restart loops

**FIXED** - GPU and diagnostic commands have timeouts, the oneshot unit has
`TimeoutStartSec=90`, restart attempts have a cooldown, and old diagnostics
are pruned.

```text
status=0 elapsed=10s
9:TimeoutStartSec=90
```

### 4. Protocol stream failure cleanup

**FIXED** - Responses and Anthropic streams stop their keepalive thread,
flush buffered content before the terminal protocol error, and close the
connection without attempting a second HTTP response.

```text
test_protocol_streams_emit_errors_and_finish ... ok
Ran 1 test in 0.542s
OK
```

### 5. Streaming tool calls inside reasoning

**FIXED** - Streaming chat accumulates tool syntax from the post-reasoning
content callback, matching the non-streaming path.

```text
test_streaming_chat_ignores_tool_calls_inside_reasoning ... ok
```

### 6. Anthropic `x-api-key`

**FIXED** - Authentication accepts `x-api-key` through constant-time
comparison while retaining Bearer support.

```text
test_anthropic_accepts_x_api_key ... ok
```

### 7. Keyless wildcard CORS

**FIXED** - The example service configuration now permits only the local
development origin instead of `*`.

```text
COLI_CORS_ORIGIN=http://localhost:5173
```

### 8. Stop targeting and pidfile safety

**FIXED** - Interactive chat engines no longer match the serve fallback,
non-Linux systems skip `/proc`, and the shared `/tmp` pidfile was removed.
This is covered by the `test_coli_cli` output under item 2.

### 9. Silent stop-set truncation

**FIXED** - Config EOS capacity increased from 8 to 64, runtime special-token
capacity increased to 256, and both limits emit warnings.

```text
[stop] warning: config.json declares 70 eos tokens; only 64 fit
oversized eos metadata -> explicit 64-token cap reached       ok
test_stops: ok
```

### 10. Responses continuity bounds and context enforcement

**FIXED** - Continuity storage is bounded by both entry count and serialized
bytes. Rendered prompts beyond the configured conservative context byte
ceiling are rejected before engine execution.

```text
test_response_history_is_bounded_by_count_and_bytes ... ok
test_rejects_prompt_over_context_byte_ceiling ... ok
```

### 11. TAP/INJECT and fused CUDA integration

**FIXED** - The reviewed TAP/INJECT branch and corrected fused-attention
branch are integrated. GB10 CUDA kernels pass, fused decode remains
byte-identical, and TAP disables PIPE2 before reading a stale host residual.

```text
[CUDA] device 0: NVIDIA GB10, 130.7 GB VRAM, sm_121
cuda backend: q8/q4/q2/f32 correctness ok on 1 device(s)
{"run": 1, "tokens": 32, "sha256": "d60a2f3cce40745b99e64ca929620de3906450abf1b170f19648c65920b9e3af"}
{"run": 2, "tokens": 32, "sha256": "d60a2f3cce40745b99e64ca929620de3906450abf1b170f19648c65920b9e3af"}
[TAP/INJECT] CUDA resident pipeline (PIPE2) disabled: hooks read/steer the host residual
tap_bytes=147480
```

### 12. Windows CUDA DLL fused symbol

**FIXED** - The fused entry point is exported, resolved, and forwarded by
`backend_loader.c`. PR #3's hosted Windows build compiled `coli_cuda.dll` with
nvcc/MSVC and linked `glm.exe` through the runtime loader.

```text
CUDA build (Windows, MSVC host)	pass	1m57s	https://github.com/anicka-net/colibri/actions/runs/29646746720/job/88086171710
CUDA syntax check	pass	29s	https://github.com/anicka-net/colibri/actions/runs/29646746720/job/88086171703
windows	pass	1m42s	https://github.com/anicka-net/colibri/actions/runs/29646746715/job/88086171647
```

### 13. Lower-severity objective defects

**FIXED** - Mixed Anthropic blocks preserve order, nameless Responses
function calls are rejected, `/profile` requires auth, zero-target OLMoE PPL
exits cleanly, and hidden aliases reuse the registry.

```text
zero-target PPL guard: ok
test_anthropic_mixed_blocks_preserve_order ... ok
test_responses_function_call_requires_name ... ok
test_profile_requires_auth_and_reports_recent_turns ... ok
```

Oversized config allocation handling, 64-node NUMA masking, and the Nix
expression are **CHANGED-UNVERIFIED** locally: allocation failure and a
64-node host are unavailable, and `nix` is not installed.

```text
nix unavailable
```

### Full validation

```text
Ran 91 tests in 3.531s
OK
```

### Self-check

1. Allocation failure, a 64-node NUMA host, and the Nix expression were not
   executable locally; each is identified above as CHANGED-UNVERIFIED. Windows
   CUDA DLL linkage was verified by PR #3's hosted MSVC job.
2. Every claim above corresponds to the `origin/dev..HEAD` diff or to pasted
   command output.
3. The review smoke paths were greedy NaN selection, misleading stop
   command lines, hung watchdog commands, engine failure after SSE headers,
   tool syntax inside reasoning, Anthropic key auth, stop overflow,
   continuity overflow, CUDA output identity, and TAP under PIPE2. Their
   outputs appear in items 1-12 and the full-suite block.

## Round 3 - DSA increment 2 safety

### 1. CUDA DSA shadow memory budget

**FIXED** - The RAM planner includes the per-slot device `Ic` shadow for every
FULL DSA layer when PIPE2 assigns it to an integrated CUDA device. The
arithmetic test covers one and three context slots at 131072 tokens, and the
eligibility test excludes SHARED indexers, dense layers, unavailable resident
paths, and discrete CUDA devices. The GB10 backend reports integrated storage.

```text
OK kv_alloc re-allocation
[CUDA] device 0: NVIDIA GB10, 130.7 GB VRAM, sm_121
cuda backend: q8/q4/q2/f32 correctness ok on 1 device(s), integrated=1
```

The complete C and Python suites also ran:

```text
test_dsa_select: 129 cases run, 0 failure(s)
test_dsa_select: ok
Ran 91 tests in 3.383s

OK
```

### 2. Versioned CUDA attention-chain ABI

**CHANGED-UNVERIFIED** - The changed export, declaration, loader resolution,
wrapper, and caller now use `coli_cuda_pipe_attn_chain_v2`. The CUDA build and
exported object symbol were checked on GB10:

```text
GB10 CUDA build: ok
000000000000d890 T coli_cuda_pipe_attn_chain_v2
```

An old-host/new-DLL and new-host/old-DLL runtime matrix requires Windows with
both DLL generations and was not available locally.

### Self-check

1. Item 2's Windows old/new DLL mismatch paths were not run because no
   nvcc/MSVC Windows host with both DLL generations was available.
2. Every claim above corresponds to the final diff or a pasted output block.
3. The review smoke paths were DSA shadow-budget arithmetic, full regression
   tests, GB10 CUDA compilation, exported-symbol inspection, and both Windows
   DLL mismatch directions. The first four outputs are above; the mismatch
   matrix is explicitly CHANGED-UNVERIFIED.

## Round 4 - Native server lifecycle

### 1. SIGPIPE terminates the native runtime

**FIXED** - `coli-native` ignores `SIGPIPE` before dispatching any command, so
socket and engine-pipe closures reach the existing write-error handling.

The fake engine was terminated before a request. The request returned HTTP
500 and the same standalone server remained healthy:

```text
500
{"status":"ok","scheduler":{"active":0,"queued":0,"capacity":1,"max_queue":8,"queue_timeout_seconds":300,"admitted":1,"completed":1,"rejected":0,"timed_out":0,"cancelled":0},"kv_slots":1,"watchdog_active":0,"tiers":{"vram":2,"ram":3,"disk":4,"vram_gb":5.00,"ram_gb":6.00},"hwinfo":{"cores":8,"ram_total_gb":16.0,"ram_avail_gb":12.0,"gpus":1,"vram_total_gb":24.0,"cpu":"Test CPU ","gpu":" Test GPU"}}
```

### 2. Failed submissions retain stack requests

**FIXED** - A failed `SUBMIT` removes its request from the pending list.
Dispatcher shutdown detaches the complete pending list before waking callers,
and every initialized error-path request now releases its buffer, condition
variable, and mutex. The regression fake engine exits during a request; two
consecutive requests return HTTP 500 while `/health` remains available.

Focused native suite:

```text
................
----------------------------------------------------------------------
Ran 16 tests in 0.599s

OK
```

The same lifecycle suite under AddressSanitizer with leak detection:

```text
................
----------------------------------------------------------------------
Ran 16 tests in 0.716s

OK
```

### 3. Native test dependency and platform wiring

**FIXED** - POSIX test discovery now builds `coli-native` and
the fake engine before either test class runs. Windows skips this POSIX-only
module, and the generic test target no longer runs the native suite twice.
Linux standalone discovery and the complete check pass:

```text
----------------------------------------------------------------------
Ran 110 tests in 6.004s

OK
```

```text
test_dsa_select: 129 cases run, 0 failure(s)
test_dsa_select: ok
----------------------------------------------------------------------
Ran 110 tests in 6.035s

OK
```

Hosted platform evidence:

```text
linux	pass	1m0s	https://github.com/anicka-net/colibri/actions/runs/29734505663/job/88326520334
macos	pass	1m34s	https://github.com/anicka-net/colibri/actions/runs/29734505663/job/88326520348
windows	pass	2m13s	https://github.com/anicka-net/colibri/actions/runs/29734505663/job/88326520394
Python tests	pass	10s	https://github.com/anicka-net/colibri/actions/runs/29734505694/job/88326520482
```

### Self-check

1. No review item remains unverified; the Windows skip path ran in hosted CI.
2. Every claim above corresponds to the current diff or a pasted command
   output.
3. The review smoke paths were engine-pipe closure, repeated submission after
   engine death, sanitizer execution, standalone Python discovery, full
   checks, and Windows test discovery. Local and hosted outputs are above.

## Round 5 - Long-context and live-VRAM safety

### 1. Device-resident expert scatter ordering

**FIXED** - The device scatter records an event on the non-blocking expert
stream and makes the default stream wait before the caller can consume or
rewrite the residual. Event setup fails before the scatter; an event/wait
failure falls back to synchronizing the expert stream.

The GB10 stress case queued 64 device groups before a default-stream overwrite:

```text
[CUDA] device 0: NVIDIA GB10, 130.7 GB VRAM, sm_121
device group ordering: ok
```

The repository's broader CUDA test still fails later on this GB10. The exact
unmodified `66f2111` test fails too, before these fixes:

```text
[CUDA] device 0: NVIDIA GB10, 130.7 GB VRAM, sm_121
mismatch 0: got 0.000000 want 0.365529
make: *** [Makefile:274: cuda-test] Error 1
baseline_rc=2
```

### 2. Integrated-memory CUDA expert accounting

**FIXED** - RAM planning now charges actual CUDA expert tensor bytes only for
integrated devices. Discrete VRAM remains outside the system-RAM budget.

The real GB10 model started with its 30 GB expert tier and the planner reported
that charge explicitly:

```text
[CUDA] hot expert tier: 1586/1586 experts, VRAM 30.00 GB (total budget 30.0 GB)
[RAM_GB=108.3 auto] resident 10.6 GB + reserve 75.2 GB (ws 1.2, KV 1x131072 25.3, CUDA experts 30.0, kvb 15.0), experts 18.9 MB x 77 layers -> cap lowered 64->15 (projected peak 107.6 GB)
```

### 3. Live repin overlap with `PILOT_REAL`

**FIXED** - Every `repin_adapt()` drains pilot work and holds `g_pilot_mx`
while scanning or mutating pin slots. The contention probe holds that mutex,
confirms repin cannot finish, releases it, and confirms repin releases the
mutex after completion.

```text
[RAM-GUARD] RSS 3.0 GB over the 1.0 GB budget: dropped 1 cached experts, cap 2 -> 1
OK kv_alloc re-allocation
```

### 4. Anthropic system-text stripping

**FIXED** - Header removal now requires the exact Claude billing envelope and
applies only to the first system block. User-authored `Authorization:` and
`x-*` content is retained.

```text
test_anthropic_transport_headers_are_not_rendered_to_model ... ok
test_anthropic_authored_system_text_is_preserved ... ok

----------------------------------------------------------------------
Ran 2 tests in 2.416s

OK
```

### 5. KV token decoder malformed input and allocation failure

**FIXED** - The tool validates numeric ranges, checkpoint length, decode-size
bounds, and both allocations before reading or decoding. A constrained address
space forces the token-ID allocation failure path.

```text
test_rejects_empty_range_argument ... ok
test_rejects_truncated_checkpoint_before_allocating ... ok
test_reports_id_allocation_failure ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.121s

OK
```

### 6. Tekton benchmark applicability

**NOT DONE** - A 2xH100 PIPE2 result cannot establish GB10 PIPE0 throughput.
This round ran a GB10 correctness/startup probe only; it makes no performance
claim and leaves production on the previous known-good build.

### Full validation

```text
test_st_pread: chunk loop + honest truncation error: ok
test_grammar: ok
expert VRAM per-device budget: ok
test_i4_grouped: ok
test_stops: ok
test_topp: ok
test_sample_nan: ok
OK kv_alloc re-allocation
test_dsa_select: ok
zero-target PPL guard: ok
test_uring: ok
Ran 120 tests in 6.791s
OK
```

Production recovery after the isolated GB10 probes:

```text
{"status":"ok","scheduler":{"active":0,"queued":0,"capacity":1,"max_queue":8,"queue_timeout_seconds":300.0,"admitted":0,"completed":0,"rejected":0,"timed_out":0,"cancelled":0},"kv_slots":1,"watchdog_active":0,"tiers":{"vram":1586,"ram":0,"disk":17870,"vram_gb":30.0,"ram_gb":0.0},"hwinfo":{"cores":20,"ram_total_gb":127.6,"ram_avail_gb":81.4,"gpus":1,"vram_total_gb":130.7,"cpu":"","gpu":"CUDA device x1"}}
```

### Self-check

1. The Tekton-to-GB10 throughput claim was not tested because it requires a
   controlled performance experiment, not a correctness patch. The Windows
   `_fseeki64` branch of the decoder was compiled only indirectly, not run.
2. Every FIXED claim above corresponds to the final diff and an executed output
   block. The benchmark item is explicitly NOT DONE.
3. The review smoke paths were CUDA residual ordering, integrated-memory
   charging, pilot/repin exclusion, transport-header removal plus authored
   system preservation, malformed/OOM token checkpoints, and the full suite.
   Their outputs appear in items 1-5 and Full validation.

## Round 6 - TC gather, streaming IDs, and Ollama metadata

### 1. TC gather result overwritten by scalar absorption

**FIXED** - Successful decode TC gather now bypasses scalar absorption and
projects the gathered context with the cached GEMV.  The scalar fallback checks
its own launch status; the H100 WMMA stage-boundary clear remains only on the
successful TC branch.

The corrected GB10 path generated the same first 32 tokens with TC disabled and
enabled, while exercising every eligible TC row:

```text
fdd8ca4062502590f49ab0095b41735e143ef98f095ffb8987d9264b16e1d8bc  /tmp/tc-off.tokens
fdd8ca4062502590f49ab0095b41735e143ef98f095ffb8987d9264b16e1d8bc  /tmp/tc-on.tokens
[PROF] DSA decode TC gather: 0 row | 0 fallback
[PROF] resident dense layers: 96 engaged | 0 fallback
[PROF] DSA decode TC gather: 2418 row | 0 fallback
[PROF] resident dense layers: 96 engaged | 0 fallback
```

The 128-token performance control engaged 9,906 TC rows with no fallback and
reduced attention and decode p50, but total wall time remained below the
production gate:

```text
tc-off: attention 48.806s | p50 846.1 ms | wall_seconds=376.03
tc-on:  attention 37.372s | p50 791.5 ms | wall_seconds=371.50
[PROF] DSA decode TC gather: 9906 row | 0 fallback
[PROF] resident dense layers: 384 engaged | 0 fallback
```

The broader CUDA test reached the new cached-GEMV and dense-resident checks,
then hit the repository's known later GB10 baseline failure:

```text
[CUDA] device 0: NVIDIA GB10, 130.7 GB VRAM, sm_121
cached q4 GEMV parity: ok
device group ordering: ok
resident dense MLP composition: ok
make: *** [Makefile:274: cuda-test] Error 1
```

### 2. Duplicate IDs across streamed tool calls

**FIXED** - Each semantic stream now owns one request-stable ID seed and assigns
the running tool index before emission.  Responses API deltas and the final
`response.completed` object use the same IDs.

### 3. Native metadata for a symlinked model root

**FIXED** - The native metadata walk follows the top-level model path, then
retains `lstat()` for recursive entries so nested symlinks cannot create loops.
The native server fixture now starts from a symlink and still reports the
expected 123-byte model and timestamp.

### 4. Python metadata for a regular-file model

**FIXED** - The Python server handles a regular-file `model_path` directly,
including its size and modification time, while directory models keep the
existing filtered walk.

The three reviewed server failure paths pass together:

```text
test_anthropic_responses_and_ollama (tests.test_native_server.NativeServerTest.test_anthropic_responses_and_ollama) ... ok
test_multiple_streamed_tool_calls_have_unique_stable_ids (tests.test_native_server.NativeServerTest.test_multiple_streamed_tool_calls_have_unique_stable_ids) ... ok
test_regular_file_model_metadata (tests.test_openai_server.ProtocolTest.test_regular_file_model_metadata) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.169s

OK
```

### Full validation

```text
.[api] tool-calls: 1 total, 1 strict, 0 de-mangled [CLEAN]
.................
----------------------------------------------------------------------
Ran 124 tests in 8.932s

OK
```

Production recovery after the isolated GB10 benchmarks:

```text
active
VmRSS:	   25424 kB
VmSwap:	       0 kB
{"status":"ok","scheduler":{"active":0,"queued":0,"capacity":1,"max_queue":8,"queue_timeout_seconds":300.0,"admitted":0,"completed":0,"rejected":0,"timed_out":0,"cancelled":0},"kv_slots":1,"watchdog_active":0,"tiers":{"vram":1586,"ram":0,"disk":17870,"vram_gb":30.0,"ram_gb":0.0},"hwinfo":{"cores":20,"ram_total_gb":127.6,"ram_avail_gb":80.7,"gpus":1,"vram_total_gb":130.7,"cpu":"","gpu":"CUDA device x1"}}
```

### Self-check

1. No review item in this round remains unverified.  The pre-fix H100
   performance result was not repeated and is marked provisional in the
   performance ledger.
2. Every claim above corresponds to the final diff or a pasted output block.
3. The reviews had no separate smoke-test list.  This round exercised the
   full-model TC branch and control, multiple streamed calls across three API
   shapes, native symlink metadata, Python regular-file metadata, the portable
   full suite, and production recovery; their outputs appear above.
