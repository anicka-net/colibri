#!/usr/bin/env python3
"""Build faithful and compact Colibri snapshots from a ModelOpt NVFP4 tree.

Routed experts remain packed E2M1. Their E4M3 group-16 scales are numerically
decoded for validation and rearranged into the CUTLASS 4.5.1 SM1xx SF atom
offline. Expert shards are written once and hard-linked into both variants.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import collections
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import nvfp4_format as nf
from convert_fp8_to_int4 import classify, dequant, quant_int8


EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
META_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")


def is_routed_expert(name: str, n_layers: int):
    match = EXPERT_RE.match(name)
    return match if match and int(match.group(1)) < n_layers else None


def dependencies():
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit("native NVFP4 conversion requires torch and safetensors") from exc
    return torch, safe_open, save_file


def raw_tensor(tensor, dtype: str | None = None) -> nf.TensorBytes:
    torch, _, _ = dependencies()
    value = tensor.detach().cpu().contiguous()
    if dtype == "BF16" or value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy()
        return nf.TensorBytes("BF16", tuple(value.shape), raw.tobytes())
    if value.dtype == torch.float8_e4m3fn:
        raw = value.view(torch.uint8).numpy()
        return nf.TensorBytes("U8", tuple(value.shape), raw.tobytes())
    return nf.tensor_bytes(value.numpy(), dtype)


def native_projection(source, name: str, keys: set[str]) -> list[tuple[str, nf.TensorBytes]]:
    torch, _, _ = dependencies()
    scale_name, tensor_name = name + "_scale", name + "_scale_2"
    input_names = (name[:-len(".weight")] + ".input_scale",
                   name + "_input_scale", name + ".input_scale")
    input_name = next((candidate for candidate in input_names if candidate in keys), input_names[0])
    required = (scale_name, tensor_name, input_name)
    missing = [key for key in required if key not in keys]
    if missing:
        raise ValueError(f"{name}: missing native ModelOpt sidecars {missing}")
    packed = source.get_tensor(name).detach().cpu().contiguous()
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError(f"{name}: expected rank-2 packed U8 E2M1 weights")
    O, half_I = packed.shape; I = half_I * 2
    scale_tensor = source.get_tensor(scale_name).detach().cpu().contiguous()
    if scale_tensor.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name}: block scales must be FP8 E4M3FN")
    raw_scale = scale_tensor.view(torch.uint8).numpy()
    if raw_scale.shape != (O, (I + 15) // 16):
        raise ValueError(f"{name}: invalid ModelOpt scale layout {raw_scale.shape}")
    decoded = nf.decode_e4m3fn(raw_scale)
    if np.any(decoded <= 0) or not np.all(np.isfinite(decoded)):
        raise ValueError(f"{name}: block scales must be positive and finite")
    tensor_scale = float(source.get_tensor(tensor_name).float().item())
    input_scale = float(source.get_tensor(input_name).float().item())
    # Exercise the existing ModelOpt dequantizer as the conversion oracle.
    reference = dequant(source, name, keys)
    native = nf.dequantize_modelopt(packed.numpy(), raw_scale, tensor_scale, I)
    if not np.array_equal(reference, native):
        delta = float(np.max(np.abs(reference - native)))
        raise ValueError(f"{name}: native payload differs from ModelOpt dequantizer ({delta})")
    swizzled = nf.swizzle_scales_for_cutlass(raw_scale, O, I)
    return [
        (name, raw_tensor(packed)),
        (name + ".nvfp4_scale", nf.tensor_bytes(swizzled)),
        (name + ".nvfp4_tensor_scale", nf.tensor_bytes(np.asarray([tensor_scale], np.float32))),
        (name + ".nvfp4_input_scale", nf.tensor_bytes(np.asarray([input_scale], np.float32))),
    ]


def convert_shard(path: pathlib.Path, n_layers: int):
    torch, safe_open, _ = dependencies()
    records: dict[tuple[int, int], list[tuple[str, nf.TensorBytes]]] = {}
    faithful, compact = {}, {}
    with safe_open(path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name in source.keys():
            match = is_routed_expert(name, n_layers)
            if match:
                key = (int(match.group(1)), int(match.group(2)))
                records.setdefault(key, []).extend(native_projection(source, name, keys))
                continue
            if name.endswith(("_scale", "_scale_2", ".input_scale", "_input_scale", "_scale_inv")):
                continue
            kind = classify(name, n_layers)
            if kind in ("skip", "consumed", "x"):
                continue
            value = source.get_tensor(name).detach().cpu()
            if kind == "f32" or value.ndim != 2:
                faithful[name] = value.float().contiguous()
                compact[name] = value.float().contiguous()
                continue
            # Faithful resident matrices are BF16. FP8 source matrices are
            # numerically dequantized before the BF16 cast; no byte reinterpretation.
            full = torch.from_numpy(dequant(source, name, keys)).to(torch.bfloat16)
            faithful[name] = full
            q, scale = quant_int8(full.float().numpy(), 8)
            compact[name] = torch.from_numpy(q.view(np.int8).copy())
            compact[name + ".qs"] = torch.from_numpy(scale)
    return records, faithful, compact


def merge_expert_records(pending, incoming):
    """Merge shard fragments and return expert records completed this shard."""
    complete = []
    for key in sorted(incoming):
        records = pending.setdefault(key, [])
        names = {name for name, _ in records}
        duplicate = names.intersection(name for name, _ in incoming[key])
        if duplicate:
            raise ValueError(f"duplicate tensors for expert {key}: {sorted(duplicate)}")
        records.extend(incoming[key])
        present = {name.split(".")[-2] for name, _ in records if name.endswith(".weight")}
        if present == {"gate_proj", "up_proj", "down_proj"}:
            complete.append(records)
            del pending[key]
        elif not present.issubset({"gate_proj", "up_proj", "down_proj"}):
            raise ValueError(f"invalid projections for expert {key}: {sorted(present)}")
    return complete


def link_exact(source: pathlib.Path, destination: pathlib.Path) -> None:
    if destination.exists():
        if os.path.samefile(source, destination): return
        raise FileExistsError(f"refusing to replace non-linked payload {destination}")
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True,
                        help="exact immutable source commit, never a branch name")
    parser.add_argument("--n-layers", type=int, default=78)
    parser.add_argument("--expected-shards", type=int, default=0,
                        help="refuse partial input until exactly this many model shards exist")
    parser.add_argument("--workers", type=int, default=1,
                        help="bounded concurrent shard conversions (results are committed in order)")
    parser.add_argument("--index-offset", type=int, default=0,
                        help="global output-shard index added to this input batch")
    args = parser.parse_args()
    source_shards = sorted(args.indir.glob("*.safetensors"))
    if not source_shards: raise SystemExit(f"no Safetensors shards in {args.indir}")
    if args.expected_shards and len(source_shards) != args.expected_shards:
        raise SystemExit(f"expected {args.expected_shards} Safetensors shards, found {len(source_shards)}")
    if args.workers < 1: raise SystemExit("--workers must be positive")
    shared=args.outdir/"shared-experts"; faithful=args.outdir/"faithful"; compact=args.outdir/"compact"
    for directory in (shared,faithful,compact): directory.mkdir(parents=True,exist_ok=True)
    _, _, save_file = dependencies()
    pending = {}; jobs = collections.deque(); source_iter = iter(enumerate(source_shards))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
      for _ in range(args.workers):
        item = next(source_iter, None)
        if item is not None:
            index, shard = item
            jobs.append((index, shard, executor.submit(convert_shard, shard, args.n_layers)))
      while jobs:
        index, shard, future = jobs.popleft()
        fragments, faithful_tensors, compact_tensors = future.result()
        expert_records = merge_expert_records(pending, fragments)
        if expert_records:
            payload=shared/f"experts-{index+args.index_offset:05d}.safetensors"
            offsets=nf.write_aligned_safetensors(payload,expert_records)
            if any(offset%4096 for offset in offsets.values()): raise AssertionError("unaligned expert record")
            link_exact(payload,faithful/payload.name); link_exact(payload,compact/payload.name)
        if faithful_tensors:
            save_file(faithful_tensors,faithful/f"resident-{index+args.index_offset:05d}.safetensors")
        if compact_tensors:
            save_file(compact_tensors,compact/f"resident-{index+args.index_offset:05d}.safetensors")
        print(f"[{index+1}/{len(source_shards)}] {shard.name}: {len(expert_records)} experts")
        item = next(source_iter, None)
        if item is not None:
            next_index, next_shard = item
            jobs.append((next_index, next_shard,
                         executor.submit(convert_shard, next_shard, args.n_layers)))
    if pending:
        detail = {key: sorted(name.split(".")[-2] for name, _ in records if name.endswith(".weight"))
                  for key, records in pending.items()}
        raise ValueError(f"incomplete experts after final shard: {detail}")
    for variant,precision in ((faithful,nf.FORMAT_BF16),(compact,nf.FORMAT_INT8_ROW)):
        manifest=nf.make_manifest(args.source_repository,args.source_revision,precision)
        (variant/nf.MANIFEST_NAME).write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        for filename in META_FILES:
            src=args.indir/filename
            if src.exists(): shutil.copy2(src,variant/filename)
    print(f"native NVFP4 snapshots written to {faithful} and {compact}")


if __name__ == "__main__": main()
