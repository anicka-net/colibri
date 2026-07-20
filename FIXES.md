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

**CHANGED-UNVERIFIED** - POSIX test discovery now builds `coli-native` and
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

The Windows skip path still needs the hosted Windows CI runner, so this item
remains CHANGED-UNVERIFIED until that job passes.

### Self-check

1. Only the Windows native-test skip path remains unverified locally because
   this host is Linux; hosted CI is the required evidence.
2. Every claim above corresponds to the current diff or a pasted command
   output.
3. The review smoke paths were engine-pipe closure, repeated submission after
   engine death, sanitizer execution, standalone Python discovery, full
   checks, and Windows test discovery. Outputs for every local path are above;
   Windows is explicitly CHANGED-UNVERIFIED.
