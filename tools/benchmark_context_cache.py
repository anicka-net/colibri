#!/usr/bin/env python3
"""Measure expert-cache residency at several context limits with one KV prefix.

The first turn prefills and checkpoints one fixed prompt. Later engine
instances restore that canonical FP32 KV checkpoint even though their maximum
context differs. Each context gets one expert-LRU warmup decode followed by an
identical measured decode.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "c"))

from openai_server import Engine, render_chat  # noqa: E402


SYSTEM = """You are a senior systems engineer auditing a production mixture-of-experts
inference runtime. Work strictly from the supplied repository evidence. Analyze
correctness, concurrency, cancellation, memory ownership, direct I/O, CUDA
synchronization, and observability. Give concrete findings and minimal patches."""

TASK_HEAD = """Audit the following Colibri implementation excerpts as one coherent
program. Identify specific defects or fragile contracts and propose a prioritized
patch series. For every proposed change, name the invariant, regression test, and
production signal. Continue for at least 1,000 words and do not conclude early.

"""

TASK_TAIL = """

Now produce the detailed audit. Cover the most consequential findings first."""

SOURCE_PATHS = (
    "c/colibri.c",
    "c/backend_cuda.cu",
    "c/native_server.c",
    "c/openai_server.py",
    "docs/serve_protocol.md",
)


def source_corpus():
    chunks = []
    for relative in SOURCE_PATHS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        chunks.append(f"\n\n===== {relative} =====\n{text}")
    return "".join(chunks)


def rendered_prompt(source):
    return render_chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK_HEAD + source + TASK_TAIL},
        ],
        enable_thinking=False,
    )


def fit_prompt(tokenizer, target):
    corpus = source_corpus()
    low, high = 0, len(corpus)
    best = None
    while low <= high:
        middle = (low + high) // 2
        prompt = rendered_prompt(corpus[:middle])
        count = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        if count <= target:
            best = (prompt, count, middle)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise RuntimeError("prompt framing alone exceeds target")
    return best


def close_engine(runtime):
    with runtime.pending_lock:
        runtime.closed = True
    if runtime.process.poll() is None:
        runtime.process.stdin.close()
        try:
            runtime.process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            runtime.process.terminate()
            try:
                runtime.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                runtime.process.kill()
                runtime.process.wait()
    runtime.dispatcher.join(timeout=2)


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def run_turn(runtime, context, phase, prompt, maximum):
    pieces = []
    started = time.monotonic()
    stats = runtime.generate(
        prompt,
        maximum,
        0.0,
        1.0,
        pieces.append,
        cache_slot=0,
    )
    wall = time.monotonic() - started
    profile = dict(runtime.profile[-1]) if runtime.profile else None
    output = "".join(pieces)
    emit(
        {
            "kind": "turn",
            "context": context,
            "phase": phase,
            "wall_s": wall,
            "stats": stats,
            "profile": profile,
            "tiers": runtime.tiers,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "output_preview": output[:160],
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=32000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=(131072, 98304, 65536, 32768),
    )
    args = parser.parse_args()

    model = Path(args.model).resolve()
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    prompt, prompt_tokens, source_chars = fit_prompt(
        tokenizer, args.target_prompt_tokens
    )
    if prompt_tokens + args.max_tokens >= min(args.contexts):
        raise SystemExit(
            f"prompt {prompt_tokens} + output {args.max_tokens} does not fit "
            f"smallest context {min(args.contexts)}"
        )
    emit(
        {
            "kind": "metadata",
            "contexts": args.contexts,
            "target_prompt_tokens": args.target_prompt_tokens,
            "tokenizer_prompt_tokens": prompt_tokens,
            "source_chars": source_chars,
            "max_tokens": args.max_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "model": str(model),
            "executable": str(Path(args.executable).resolve()),
        }
    )

    for index, context in enumerate(args.contexts):
        env = dict(os.environ)
        env.update(
            {
                "CTX": str(context),
                "COLI_CONTEXT": str(context),
                "KVSAVE": "1",
                "COLI_KV_CACHE_GB": "0",
                "AUTOPIN": "0",
                "CAP_RAISE": "0",
                "COLI_ADAPTIVE_CAP": "0",
                "PROF": "1",
                "TEMP": "0",
            }
        )
        load_started = time.monotonic()
        runtime = Engine(
            args.executable,
            model,
            cap=256,
            max_tokens=args.max_tokens,
            env=env,
            kv_slots=1,
            expert_bits=4,
            dense_bits=8,
        )
        emit(
            {
                "kind": "startup",
                "context": context,
                "load_s": time.monotonic() - load_started,
                "tiers": runtime.tiers,
                "hardware": runtime.hwinfo,
                "restored_checkpoint_expected": index > 0,
            }
        )
        try:
            run_turn(runtime, context, "warmup", prompt, args.max_tokens)
            run_turn(runtime, context, "measured", prompt, args.max_tokens)
        finally:
            close_engine(runtime)


if __name__ == "__main__":
    main()
