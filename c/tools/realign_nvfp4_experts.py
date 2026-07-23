#!/usr/bin/env python3
"""Byte-preserving rewrite of Colibri NVFP4 expert records.

This does not dequantize or otherwise reinterpret tensors. It rewrites existing
expert Safetensors with 4-KiB record starts and 16-byte component starts, then
builds faithful/compact trees whose expert shards hard-link the new payload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import errno
import json
import os
import pathlib
import re
import struct
import tempfile

import nvfp4_format as nf


EXPERT_COMPONENT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight"
    r"(\.nvfp4_(?:scale|tensor_scale|input_scale))?$"
)
EXPECTED_SUFFIXES = {
    f"{projection}{suffix}"
    for projection in ("gate_proj", "up_proj", "down_proj")
    for suffix in ("", ".nvfp4_scale", ".nvfp4_tensor_scale", ".nvfp4_input_scale")
}


def _encode_header(header: dict) -> bytes:
    header["__metadata__"] = {
        "format": "colibri-component-aligned-expert-records-v2",
        "record_alignment": "4096",
        "component_alignment": "16",
        "pad": "",
    }
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    header["__metadata__"]["pad"] = " " * ((-(8 + len(encoded))) % 4096)
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    if (8 + len(encoded)) % 4096:
        raise AssertionError("failed to align rewritten Safetensors data section")
    return encoded


def _copy_range(source, output, source_offset: int, length: int) -> None:
    source.seek(source_offset)
    remaining = length
    use_kernel_copy = hasattr(os, "copy_file_range")
    while remaining:
        amount = min(remaining, 1 << 30)
        if use_kernel_copy:
            try:
                copied = os.copy_file_range(source.fileno(), output.fileno(), amount)
            except OSError as exc:
                if exc.errno not in (errno.EXDEV, errno.EINVAL, errno.ENOSYS, errno.EPERM):
                    raise
                use_kernel_copy = False
                continue
            if copied == 0:
                raise EOFError(f"short copy at source offset {source.tell()}")
        else:
            chunk = source.read(min(remaining, 8 << 20))
            if not chunk:
                raise EOFError(f"short read at source offset {source.tell()}")
            copied = output.write(chunk)
            if copied != len(chunk):
                raise OSError("short output write")
        remaining -= copied


def plan_rewrite(source_path: pathlib.Path):
    data_start, source_header = nf.read_safetensors_header(source_path)
    records: dict[tuple[int, int], list[tuple[int, str, dict]]] = {}
    for name, metadata in source_header.items():
        if name == "__metadata__" or name.startswith("__coli_"):
            continue
        match = EXPERT_COMPONENT.match(name)
        if not match:
            raise ValueError(f"{source_path}: unexpected non-expert tensor {name}")
        begin, end = metadata["data_offsets"]
        if begin < 0 or end < begin:
            raise ValueError(f"{source_path}: invalid offsets for {name}")
        key = int(match.group(1)), int(match.group(2))
        records.setdefault(key, []).append((data_start + begin, name, metadata))
    if not records:
        raise ValueError(f"{source_path}: no expert records")

    output_header: dict[str, dict] = {}
    chunks: list[tuple[int | None, int]] = []
    cursor = 0
    for record_number, (key, components) in enumerate(
            sorted(records.items(), key=lambda item: min(part[0] for part in item[1]))):
        suffixes = {
            f"{EXPERT_COMPONENT.match(name).group(3)}"
            f"{EXPERT_COMPONENT.match(name).group(4) or ''}"
            for _, name, _ in components
        }
        if suffixes != EXPECTED_SUFFIXES:
            raise ValueError(f"{source_path}: incomplete expert {key}: {sorted(suffixes)}")
        padding = (-cursor) % 4096
        if padding:
            name = f"__coli_padding_{record_number:08d}"
            output_header[name] = {"dtype": "U8", "shape": [padding],
                                   "data_offsets": [cursor, cursor + padding]}
            chunks.append((None, padding)); cursor += padding
        for component_number, (absolute, name, metadata) in enumerate(sorted(components)):
            padding = (-cursor) % 16
            if padding:
                pad_name = f"__coli_component_padding_{record_number:08d}_{component_number:02d}"
                output_header[pad_name] = {"dtype": "U8", "shape": [padding],
                                           "data_offsets": [cursor, cursor + padding]}
                chunks.append((None, padding)); cursor += padding
            begin, end = metadata["data_offsets"]; length = end - begin
            output_header[name] = {
                "dtype": metadata["dtype"], "shape": metadata["shape"],
                "data_offsets": [cursor, cursor + length],
            }
            chunks.append((absolute, length)); cursor += length
    return _encode_header(output_header), chunks, len(records)


def rewrite_shard(source_path: pathlib.Path, target_path: pathlib.Path) -> tuple[int, int, int]:
    encoded, chunks, records = plan_rewrite(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=target_path.name + ".", suffix=".tmp", dir=target_path.parent
    )
    try:
        with open(source_path, "rb", buffering=0) as source, os.fdopen(fd, "wb", buffering=0) as out:
            out.write(struct.pack("<Q", len(encoded))); out.write(encoded)
            for source_offset, length in chunks:
                if source_offset is None:
                    out.write(bytes(length))
                else:
                    _copy_range(source, out, source_offset, length)
            os.fsync(out.fileno())
        os.replace(temporary, target_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return records, source_path.stat().st_size, target_path.stat().st_size


def verify_shard_samples(source_path: pathlib.Path, target_path: pathlib.Path,
                         sample_bytes: int = 64) -> int:
    """Verify metadata, alignment and both ends of every logical tensor."""
    if sample_bytes < 1:
        raise ValueError("sample_bytes must be positive")
    source_base, source_header = nf.read_safetensors_header(source_path)
    target_base, target_header = nf.read_safetensors_header(target_path)
    source_names = {
        name for name in source_header
        if name != "__metadata__" and not name.startswith("__coli_")
    }
    target_names = {
        name for name in target_header
        if name != "__metadata__" and not name.startswith("__coli_")
    }
    if source_names != target_names:
        raise ValueError("rewritten shard tensor-name set differs from source")
    with open(source_path, "rb") as source, open(target_path, "rb") as target:
        for name in source_names:
            old, new = source_header[name], target_header[name]
            if old["dtype"] != new["dtype"] or old["shape"] != new["shape"]:
                raise ValueError(f"{name}: rewritten dtype/shape differs")
            old_begin, old_end = old["data_offsets"]
            new_begin, new_end = new["data_offsets"]
            size = old_end - old_begin
            if size != new_end - new_begin or (target_base + new_begin) % 16:
                raise ValueError(f"{name}: rewritten length/alignment differs")
            count = min(sample_bytes, size)
            for delta in {0, max(0, size - count)}:
                source.seek(source_base + old_begin + delta)
                target.seek(target_base + new_begin + delta)
                if source.read(count) != target.read(count):
                    raise ValueError(f"{name}: rewritten sample differs")
    return len(source_names)


def _link_variant(source: pathlib.Path, destination: pathlib.Path,
                  payload: pathlib.Path, expert_names: set[str]) -> None:
    destination.mkdir()
    for item in source.iterdir():
        if item.name in expert_names:
            continue
        target = destination / item.name
        if item.name == nf.MANIFEST_NAME:
            manifest = nf.load_manifest(source)
            if manifest is None:
                raise ValueError(f"{source}: missing native manifest")
            manifest["expert_record"].update({
                "component_alignment": 16, "layout": "component-aligned-v2",
            })
            target.write_text(json.dumps(nf.validate_manifest(manifest),
                                         indent=2, sort_keys=True) + "\n")
        elif item.is_file():
            os.link(item, target)
    for name in expert_names:
        os.link(payload / name, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=pathlib.Path, required=True,
                        help="tree containing shared-experts, faithful and compact")
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.out_root.exists():
        raise SystemExit(f"refusing to replace existing output {args.out_root}")
    source_payload = args.snapshot_root / "shared-experts"
    shards = sorted(source_payload.glob("experts-*.safetensors"))
    if not shards:
        raise SystemExit(f"no expert shards in {source_payload}")
    output_payload = args.out_root / "shared-experts"
    output_payload.mkdir(parents=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs = [(shard, executor.submit(
                rewrite_shard, shard, output_payload / shard.name)) for shard in shards]
            total_records = total_source = total_output = 0
            for index, (shard, future) in enumerate(jobs, 1):
                records, source_bytes, output_bytes = future.result()
                total_records += records; total_source += source_bytes; total_output += output_bytes
                print(f"[{index}/{len(shards)}] {shard.name}: {records} records, "
                      f"{source_bytes / (1 << 30):.2f} -> {output_bytes / (1 << 30):.2f} GiB",
                      flush=True)
        names = {shard.name for shard in shards}
        _link_variant(args.snapshot_root / "faithful", args.out_root / "faithful",
                      output_payload, names)
        _link_variant(args.snapshot_root / "compact", args.out_root / "compact",
                      output_payload, names)
        print(f"rewrote {total_records} immutable records: "
              f"{total_source / (1 << 30):.2f} -> {total_output / (1 << 30):.2f} GiB",
              flush=True)
    except BaseException:
        print(f"incomplete output retained for diagnosis: {args.out_root}")
        raise


if __name__ == "__main__":
    main()
