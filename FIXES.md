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

**CHANGED-UNVERIFIED** - The fused entry point is exported, resolved, and
forwarded by `backend_loader.c`. No MinGW/MSVC CUDA toolchain is installed on
this machine; the pull request's Windows CUDA job is the executable proof.

```text
CUDA DLL loader exports, resolves, and forwards pipe_attn_chain
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

1. Windows CUDA DLL linkage, allocation failure, a 64-node NUMA host, and the
   Nix expression were not executable locally; each is identified above as
   CHANGED-UNVERIFIED.
2. Every claim above corresponds to the `origin/dev..HEAD` diff or to pasted
   command output.
3. The review smoke paths were greedy NaN selection, misleading stop
   command lines, hung watchdog commands, engine failure after SSE headers,
   tool syntax inside reasoning, Anthropic key auth, stop overflow,
   continuity overflow, CUDA output identity, and TAP under PIPE2. Their
   outputs appear in items 1-12 and the full-suite block.
