#!/usr/bin/env python3
"""Benchmark Colibri as a persistent single-user service.

Unlike the engine's end-to-end STAT rate, this driver separates time to first
token from the subsequent decode interval.  Every turn changes an early nonce,
so warmups heat the expert tiers without letting a measured prompt reuse more
than the chat-template prefix. Results are emitted as JSON lines for
straightforward archival and comparison.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "c"))

from openai_server import Engine, render_chat  # noqa: E402

PROFILE_ENV_KEYS = (
    "TEMP",
    "CTX",
    "COLI_KV_DTYPE",
    "COLI_CUDA",
    "CUDA_DENSE",
    "COLI_CUDA_HOST_EXPERTS",
    "COLI_CUDA_PIPE",
    "COLI_CUDA_PIPE_S_MIN",
    "COLI_CUDA_PREFILL",
    "CUDA_EXPERT_GB",
    "PIN",
    "PIN_GB",
    "AUTOPIN",
    "DIRECT",
    "URING",
    "PIPE",
    "PIPE_WORKERS",
    "PILOT_REAL",
    "PILOT_K",
    "COUPLE",
    "COUPLE_K",
    "COUPLE_D",
    "COLI_ADAPTIVE_CAP",
    "DRAFT",
    "CACHE_ROUTE",
    "COLI_DSA_REFRESH",
    "COLI_KV_SHARE",
    "KVSAVE",
    "COLI_NVFP4_NATIVE",
    "COLI_NVFP4_NATIVE_MIN_ROWS",
)


SHORT_SYSTEM = """You are a senior software engineer reviewing production code.
Give a concrete, technically precise answer. Preserve observable behavior, call
out concurrency and error-handling risks, and prefer a small patch over a broad
rewrite."""

SHORT_USER = """Review this function and propose a corrected implementation:

```c
int read_exact(int fd, void *dst, size_t n) {
    ssize_t got = read(fd, dst, n);
    return got == (ssize_t)n ? 0 : -1;
}
```

The descriptor may be interrupted by signals or return short reads. Explain the
failure modes, then provide portable POSIX C code that handles them. Include the
important edge cases and a compact test plan."""

LONG_SYSTEM = """You are the senior engineer responsible for a production
inference service. Work from the supplied repository evidence, not generic
advice. Your review must preserve protocol compatibility and model semantics.

Treat latency, correctness, resource bounds, cancellation, durability, and
observability as equally important. Distinguish time-to-first-token from decode
throughput. Account for concurrent requests, prefix reuse, partial I/O, process
shutdown, and failures that occur after output has begun. Do not silently weaken
validation or error handling to improve a benchmark.

When proposing code, identify the precise invariants each change maintains.
Prefer a minimal sequence of independently testable patches. For every patch,
state the regression test, the operational signal that would expose failure in
production, and any compatibility constraint. Explain ambiguous evidence and
make your assumptions explicit. Finish with a prioritized implementation plan
and a deployment/rollback checklist suitable for an on-call engineer."""


def between(path, start, end):
    text = (ROOT / path).read_text(encoding="utf-8")
    first = text.index(start)
    last = text.index(end, first)
    return text[first:last]


def make_prompt(workload, nonce="colibri-throughput-00"):
    nonce_line = f"Benchmark sample nonce: {nonce}.\n"
    if workload == "short":
        messages = [
            {"role": "system", "content": nonce_line + SHORT_SYSTEM},
            {"role": "user", "content": SHORT_USER},
        ]
    else:
        protocol = (ROOT / "docs" / "serve_protocol.md").read_text(encoding="utf-8")
        mux = between(
            "c/colibri.c",
            "typedef struct {\n    int active, pending, emitted",
            "static void run_serve_mux",
        )
        user = f"""Audit the multiplexed generation path below. We have seen
field failures involving prefix reuse, cancellation, misleading throughput
telemetry, and persistence ordering. Find concrete defects or fragile
contracts, then propose a minimal patch series with tests. Pay special
attention to whether the documented wire protocol agrees with the C engine.

## Protocol specification

{protocol}

## C multiplexing implementation

```c
{mux}
```
"""
        messages = [
            {"role": "system", "content": nonce_line + LONG_SYSTEM},
            {"role": "user", "content": user},
        ]
    return render_chat(messages, enable_thinking=False)


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * quantile
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_digest():
    """Identify a dirty benchmark candidate without pretending HEAD is enough."""
    digest = hashlib.sha256()
    for relative in (
        "c/backend_cuda.cu",
        "c/colibri.c",
        "c/tests/test_backend_cuda.cu",
        "tools/benchmark_service_throughput.py",
    ):
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def graceful_close(runtime):
    """Let the engine reach its normal EOF cleanup so CUDA counters are logged."""
    with runtime.pending_lock:
        if runtime.closed:
            return
        runtime.closed = True
    if runtime.process.poll() is None:
        try:
            runtime.process.stdin.close()
            runtime.process.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            runtime.process.terminate()
            try:
                runtime.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                runtime.process.kill()
                runtime.process.wait()
    runtime.dispatcher.join(timeout=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workload", choices=("short", "long"), required=True)
    parser.add_argument("--cap", type=int, default=17)
    parser.add_argument("--expert-bits", type=int, default=4)
    parser.add_argument("--dense-bits", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    turns = args.warmups + args.samples
    prompt = make_prompt(args.workload)
    metadata = {
        "kind": "metadata",
        "label": args.label,
        "workload": args.workload,
        "commit": git_revision(),
        "source_sha256": source_digest(),
        "model": str(Path(args.model).resolve()),
        "prompt_bytes": len(prompt.encode("utf-8")),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "warmups": args.warmups,
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "cap": args.cap,
        "expert_bits": args.expert_bits,
        "dense_bits": args.dense_bits,
        "kv_slots": 1,
        "environment": {
            key: os.environ[key] for key in PROFILE_ENV_KEYS if key in os.environ
        },
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)

    load_started = time.monotonic()
    runtime = Engine(
        args.executable,
        args.model,
        cap=args.cap,
        max_tokens=args.max_tokens,
        kv_slots=1,
        expert_bits=args.expert_bits,
        dense_bits=args.dense_bits,
    )
    load_seconds = time.monotonic() - load_started
    print(
        json.dumps(
            {
                "kind": "startup",
                "label": args.label,
                "workload": args.workload,
                "load_s": load_seconds,
                "hardware": runtime.hwinfo,
                "tiers": runtime.tiers,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    measured = []
    try:
        for index in range(turns):
            phase = "warmup" if index < args.warmups else "measured"
            prompt = make_prompt(
                args.workload, f"colibri-throughput-{args.workload}-{index:02d}"
            )
            visible_times = []
            output = []
            started = time.monotonic()

            def on_text(text):
                visible_times.append(time.monotonic())
                output.append(text)

            stats = runtime.generate(
                prompt,
                args.max_tokens,
                0.0,
                1.0,
                on_text,
                cache_slot=0,
            )
            finished = time.monotonic()
            first = visible_times[0] if visible_times else finished
            last = visible_times[-1] if visible_times else finished
            ttft = first - started
            completion = stats["completion_tokens"]
            decode_tokens = max(0, completion - 1)
            decode_elapsed = max(0.0, last - first)
            decode_tps = (
                decode_tokens / decode_elapsed
                if decode_tokens > 0 and decode_elapsed > 0
                else None
            )
            intervals = [
                visible_times[n] - visible_times[n - 1]
                for n in range(1, len(visible_times))
            ]
            profile = dict(runtime.profile[-1]) if runtime.profile else None
            record = {
                "kind": "sample",
                "label": args.label,
                "workload": args.workload,
                "phase": phase,
                "index": index if phase == "warmup" else index - args.warmups,
                "slot": 0,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "wall_s": finished - started,
                "ttft_s": ttft,
                "prefill_prompt_tps": (
                    stats["prompt_tokens"] / ttft if ttft > 0 else None
                ),
                "decode_s": decode_elapsed,
                "decode_tps": decode_tps,
                "visible_interval_p50_s": percentile(intervals, 0.50),
                "visible_interval_p95_s": percentile(intervals, 0.95),
                "engine_end_to_end_output_tps": stats["tokens_per_second"],
                "cache_hit_percent": stats["cache_hit_percent"],
                "rss_gb": stats["rss_gb"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": completion,
                "length_limited": stats["length_limited"],
                "profile": profile,
                "output_sha256": hashlib.sha256(
                    "".join(output).encode("utf-8")
                ).hexdigest(),
                "output_preview": "".join(output)[:160],
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            if phase == "measured":
                measured.append(record)
    finally:
        graceful_close(runtime)

    summary = {
        "kind": "summary",
        "label": args.label,
        "workload": args.workload,
        "samples": len(measured),
        "load_s": load_seconds,
    }
    for field in (
        "wall_s",
        "ttft_s",
        "prefill_prompt_tps",
        "decode_tps",
        "cache_hit_percent",
        "rss_gb",
    ):
        values = [item[field] for item in measured if item[field] is not None]
        summary[f"median_{field}"] = statistics.median(values) if values else None
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
