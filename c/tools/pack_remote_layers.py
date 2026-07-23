#!/usr/bin/env python3
"""Pack complete routed-expert layers for the optional remote CUDA worker."""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

MAGIC = b"COLIRXP1"
VERSION = 1
HIDDEN = 6144
INTER = 2048
HEADER = struct.Struct("<8sIIIIQQ")
RECORD = struct.Struct("<HHI12Q")
FIELDS = (
    ("gate_proj.weight", 0),
    ("gate_proj.weight.qs", 1),
    ("up_proj.weight", 2),
    ("up_proj.weight.qs", 3),
    ("down_proj.weight", 4),
    ("down_proj.weight.qs", 5),
)


def parse_layers(value):
    layers = sorted({int(item) for item in value.split(",")})
    if not layers or layers[0] < 0 or layers[-1] > 127:
        raise argparse.ArgumentTypeError("layers must be comma-separated values in 0..127")
    return layers


def read_header(path):
    with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(size)), 8 + size


def copy_range(source, target, offset, size, digest, chunk_size=8 << 20):
    source.seek(offset)
    while size:
        data = source.read(min(size, chunk_size))
        if not data:
            raise OSError("unexpected EOF while reading source tensor")
        target.write(data)
        digest.update(data)
        size -= len(data)


def pack(snapshot, output, layers):
    wanted = {}
    records = []
    for layer in layers:
        for eid in range(256):
            record = {"layer": layer, "eid": eid, "tensors": [None] * 6}
            records.append(record)
            for suffix, slot in FIELDS:
                name = f"model.layers.{layer}.mlp.experts.{eid}.{suffix}"
                wanted[name] = (record, slot)

    for shard in sorted(snapshot.glob("out-*.safetensors")):
        metadata, base = read_header(shard)
        for name in wanted.keys() & metadata.keys():
            lo, hi = metadata[name]["data_offsets"]
            record, slot = wanted[name]
            record["tensors"][slot] = (shard, base + lo, hi - lo)

    missing = [
        f"L{record['layer']}/E{record['eid']}/slot{slot}"
        for record in records
        for slot, tensor in enumerate(record["tensors"])
        if tensor is None
    ]
    if missing:
        raise SystemExit(f"missing {len(missing)} tensors, first: {missing[:5]}")

    records_offset = HEADER.size
    data_offset = (records_offset + len(records) * RECORD.size + 4095) & ~4095
    cursor = data_offset
    by_shard = {}
    for record in records:
        record["offsets"] = [0] * 6
        record["sizes"] = [0] * 6
        for slot, (shard, source_offset, size) in enumerate(record["tensors"]):
            by_shard.setdefault(shard, []).append(
                (source_offset, size, record, slot)
            )
    for shard in sorted(by_shard):
        for source_offset, size, record, slot in sorted(
            by_shard[shard], key=lambda item: item[0]
        ):
            record["offsets"][slot] = cursor
            record["sizes"][slot] = size
            cursor += size

    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("wb") as file:
        header = HEADER.pack(
            MAGIC, VERSION, HIDDEN, INTER, len(records), records_offset, data_offset
        )
        file.write(header)
        digest.update(header)
        for record in records:
            packed = RECORD.pack(
                record["layer"],
                record["eid"],
                0,
                *record["offsets"],
                *record["sizes"],
            )
            file.write(packed)
            digest.update(packed)
        padding = b"\0" * (data_offset - file.tell())
        file.write(padding)
        digest.update(padding)
        for shard in sorted(by_shard):
            with shard.open("rb") as source:
                for source_offset, size, record, slot in sorted(
                    by_shard[shard], key=lambda item: item[0]
                ):
                    if file.tell() != record["offsets"][slot]:
                        raise RuntimeError("pack offset mismatch")
                    copy_range(source, file, source_offset, size, digest)
        file.flush()
        os.fsync(file.fileno())
    return len(records), output.stat().st_size, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", required=True, type=parse_layers)
    args = parser.parse_args()
    count, size, digest = pack(args.snapshot, args.output, args.layers)
    print(f"experts={count} bytes={size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
