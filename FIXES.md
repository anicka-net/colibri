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
Ran 69 tests in 1.998s

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
