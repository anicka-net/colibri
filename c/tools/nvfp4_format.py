"""Colibri's versioned ModelOpt NVFP4 snapshot contract.

This module deliberately has no Torch or safetensors dependency.  Conversion,
the C loader tests, and CUDA fixture generators use the same validation and
scale-layout code, so native tensors are never identified from byte counts.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MANIFEST_NAME = "colibri-manifest.json"
SCHEMA = "colibri.snapshot"
SCHEMA_VERSION = 1
FORMAT_BF16 = "bf16"
FORMAT_F32 = "f32"
FORMAT_INT8_ROW = "int8-row"
FORMAT_MODELOPT_NVFP4 = "modelopt-nvfp4-e2m1"
NVFP4_GROUP_SIZE = 16
MODELOPT_SCALE_LAYOUT = "modelopt-row-major-o-by-ceil-i16"
CUTLASS_SCALE_LAYOUT = "cutlass-sm1xx-sf-atom-128x4-v1"
CUTLASS_REVISION = "2e602843e75100d0e03934efb386b3e1e35d7907"  # v4.5.1

E2M1 = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class TensorBytes:
    dtype: str
    shape: tuple[int, ...]
    data: bytes


_DTYPE_BYTES = {"U8": 1, "I8": 1, "BF16": 2, "F16": 2, "F32": 4}


def tensor_bytes(array: np.ndarray, dtype: str | None = None) -> TensorBytes:
    value = np.ascontiguousarray(array)
    inferred = {np.dtype(np.uint8): "U8", np.dtype(np.int8): "I8",
                np.dtype(np.float32): "F32"}.get(value.dtype)
    kind = dtype or inferred
    if kind not in _DTYPE_BYTES:
        raise ValueError(f"unsupported snapshot tensor dtype {kind!r}")
    if value.nbytes != math.prod(value.shape) * _DTYPE_BYTES[kind]:
        raise ValueError("tensor byte length does not match dtype and shape")
    return TensorBytes(kind, tuple(int(x) for x in value.shape), value.tobytes())


def write_aligned_safetensors(
    path: os.PathLike[str] | str,
    records: list[list[tuple[str, TensorBytes]]],
    alignment: int = 4096,
) -> dict[str, int]:
    """Write standard Safetensors with an aligned start for every record.

    Padding is represented by reserved U8 tensors, keeping all data offsets
    valid for ordinary Safetensors readers. Returns absolute record offsets.
    """
    if alignment not in (4096, 16384):
        raise ValueError("record alignment must be 4096 or 16384")
    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    record_relative: dict[str, int] = {}
    cursor = 0
    seen: set[str] = set()
    for record_number, record in enumerate(records):
        if not record:
            raise ValueError("expert record cannot be empty")
        padding = (-cursor) % alignment
        if padding:
            name = f"__coli_padding_{record_number:08d}"
            header[name] = {"dtype": "U8", "shape": [padding],
                            "data_offsets": [cursor, cursor + padding]}
            chunks.append(bytes(padding)); cursor += padding
        record_relative[record[0][0]] = cursor
        for name, tensor in record:
            if name in seen or name.startswith("__coli_padding_"):
                raise ValueError(f"duplicate or reserved tensor name {name!r}")
            seen.add(name)
            expected = math.prod(tensor.shape) * _DTYPE_BYTES.get(tensor.dtype, 0)
            if expected != len(tensor.data):
                raise ValueError(f"{name}: byte length does not match dtype/shape")
            end = cursor + len(tensor.data)
            header[name] = {"dtype": tensor.dtype, "shape": list(tensor.shape),
                            "data_offsets": [cursor, end]}
            chunks.append(tensor.data); cursor = end
    header["__metadata__"] = {"format": "colibri-aligned-expert-records-v1", "pad": ""}
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    pad = (-(8 + len(encoded))) % alignment
    header["__metadata__"]["pad"] = " " * pad
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    if (8 + len(encoded)) % alignment:
        raise AssertionError("failed to align Safetensors data section")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(struct.pack("<Q", len(encoded))); out.write(encoded)
            for chunk in chunks: out.write(chunk)
            out.flush(); os.fsync(out.fileno())
        os.replace(temporary, target)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    data_start = 8 + len(encoded)
    return {name: data_start + offset for name, offset in record_relative.items()}


def read_safetensors_header(path: os.PathLike[str] | str) -> tuple[int, dict[str, Any]]:
    with open(path, "rb") as src:
        raw = src.read(8)
        if len(raw) != 8: raise ValueError("truncated Safetensors prefix")
        length = struct.unpack("<Q", raw)[0]
        if length > 512 << 20: raise ValueError("oversized Safetensors header")
        header = json.loads(src.read(length))
    return 8 + length, header


def decode_e4m3fn(raw: np.ndarray) -> np.ndarray:
    """Decode raw E4M3FN bytes, rejecting its NaN encoding."""
    b = np.asarray(raw, dtype=np.uint8)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = ((b >> 3) & 0x0F).astype(np.int16)
    mant = (b & 0x07).astype(np.int16)
    if np.any((exp == 0x0F) & (mant == 0x07)):
        raise ValueError("E4M3FN block scales contain NaN")
    normal = np.ldexp(1.0 + mant.astype(np.float32) / 8.0, exp - 7)
    subnormal = np.ldexp(mant.astype(np.float32), -9)
    return (sign * np.where(exp == 0, subnormal, normal)).astype(np.float32)


def unpack_e2m1(packed: np.ndarray, input_size: int) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != (input_size + 1) // 2:
        raise ValueError("packed E2M1 shape does not match input size")
    codes = np.empty((p.shape[0], input_size), dtype=np.uint8)
    codes[:, 0::2] = p[:, : (input_size + 1) // 2] & 0x0F
    if input_size > 1:
        codes[:, 1::2] = (p[:, : input_size // 2] >> 4) & 0x0F
    return E2M1[codes]


def dequantize_modelopt(
    packed: np.ndarray,
    block_scale_e4m3: np.ndarray,
    tensor_scale: float,
    input_size: int,
) -> np.ndarray:
    """Scalar/vectorized W4A32 oracle for one [O,I] ModelOpt tensor."""
    weights = unpack_e2m1(packed, input_size)
    raw_scale = np.asarray(block_scale_e4m3, dtype=np.uint8)
    expected = (weights.shape[0], (input_size + NVFP4_GROUP_SIZE - 1) // NVFP4_GROUP_SIZE)
    if raw_scale.shape != expected:
        raise ValueError(f"block-scale shape {raw_scale.shape} != {expected}")
    scale = decode_e4m3fn(raw_scale)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("NVFP4 block scales must be finite and positive")
    if not math.isfinite(float(tensor_scale)) or not 0.0 < float(tensor_scale) < 1.0:
        raise ValueError("ModelOpt tensor scale must be finite and in (0,1)")
    expanded = np.repeat(scale, NVFP4_GROUP_SIZE, axis=1)[:, :input_size]
    return weights * expanded * np.float32(tensor_scale)


def cutlass_scale_storage_shape(output_size: int, input_size: int) -> tuple[int, int, int]:
    """Return (M tiles, K tiles, bytes) for CUTLASS's SM1xx SF atom.

    CUTLASS 4.5.1 defines the K-major atom as logical [128,64] with
    strides ((16,4),(0,1)).  Sixteen K coordinates therefore duplicate one
    scale and four distinct FP8 scale factors occupy each row of the atom.
    """
    if output_size <= 0 or input_size <= 0:
        raise ValueError("NVFP4 dimensions must be positive")
    mt = (output_size + 127) // 128
    kt = (input_size + 63) // 64
    return mt, kt, mt * kt * 512


def swizzle_scales_for_cutlass(raw: np.ndarray, output_size: int, input_size: int) -> np.ndarray:
    """Convert ModelOpt row-major group-16 scales to the SM1xx SF atom."""
    src = np.asarray(raw, dtype=np.uint8)
    groups = (input_size + 15) // 16
    if src.shape != (output_size, groups):
        raise ValueError(f"ModelOpt scale shape {src.shape} != {(output_size, groups)}")
    mt, kt, size = cutlass_scale_storage_shape(output_size, input_size)
    dst = np.zeros(size, dtype=np.uint8)
    for o in range(output_size):
        om, oi = divmod(o, 128)
        for g in range(groups):
            kg, gi = divmod(g, 4)
            atom = (om * kt + kg) * 512
            dst[atom + (oi % 32) * 16 + (oi // 32) * 4 + gi] = src[o, g]
    return dst


def unswizzle_scales_from_cutlass(raw: np.ndarray, output_size: int, input_size: int) -> np.ndarray:
    mt, kt, size = cutlass_scale_storage_shape(output_size, input_size)
    del mt
    src = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if src.size != size:
        raise ValueError(f"CUTLASS scale storage has {src.size} bytes, expected {size}")
    groups = (input_size + 15) // 16
    dst = np.empty((output_size, groups), dtype=np.uint8)
    for o in range(output_size):
        om, oi = divmod(o, 128)
        for g in range(groups):
            kg, gi = divmod(g, 4)
            atom = (om * kt + kg) * 512
            dst[o, g] = src[atom + (oi % 32) * 16 + (oi // 32) * 4 + gi]
    return dst


def validate_manifest(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized manifest or raise before any tensor is loaded."""
    if doc.get("schema") != SCHEMA or doc.get("version") != SCHEMA_VERSION:
        raise ManifestError("unsupported Colibri snapshot manifest schema/version")
    source = doc.get("source")
    if not isinstance(source, dict) or not source.get("repository") or not source.get("revision"):
        raise ManifestError("manifest requires source repository and exact revision")
    resident = doc.get("resident_precision")
    if resident not in (FORMAT_BF16, FORMAT_INT8_ROW):
        raise ManifestError("resident_precision must be bf16 or int8-row")
    expert = doc.get("routed_experts")
    if not isinstance(expert, dict) or expert.get("format") != FORMAT_MODELOPT_NVFP4:
        raise ManifestError("routed experts must explicitly declare ModelOpt NVFP4")
    required = {
        "group_size": NVFP4_GROUP_SIZE,
        "weight_layout": "e2m1-low-nibble-even",
        "source_scale_layout": MODELOPT_SCALE_LAYOUT,
        "scale_layout": CUTLASS_SCALE_LAYOUT,
        "scale_dtype": "fp8-e4m3fn",
        "tensor_scale_dtype": FORMAT_F32,
        "input_scale_dtype": FORMAT_F32,
    }
    for key, value in required.items():
        if expert.get(key) != value:
            raise ManifestError(f"unsupported routed_experts.{key}: {expert.get(key)!r}")
    cutlass = doc.get("cutlass")
    if not isinstance(cutlass, dict) or cutlass.get("version") != "4.5.1" or cutlass.get("revision") != CUTLASS_REVISION:
        raise ManifestError("snapshot requires pinned CUTLASS 4.5.1 revision")
    if doc.get("expert_record", {}).get("alignment") not in (4096, 16384):
        raise ManifestError("expert record alignment must be 4096 or 16384")
    return dict(doc)


def load_manifest(snapshot: os.PathLike[str] | str) -> dict[str, Any] | None:
    path = Path(snapshot) / MANIFEST_NAME
    if not path.exists():
        return None  # legacy snapshots retain their existing inferred format path
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ManifestError("snapshot manifest root must be an object")
    return validate_manifest(doc)


def make_manifest(repository: str, revision: str, resident_precision: str) -> dict[str, Any]:
    return validate_manifest({
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "source": {"repository": repository, "revision": revision},
        "resident_precision": resident_precision,
        "routed_experts": {
            "format": FORMAT_MODELOPT_NVFP4,
            "group_size": NVFP4_GROUP_SIZE,
            "weight_layout": "e2m1-low-nibble-even",
            "source_scale_layout": MODELOPT_SCALE_LAYOUT,
            "scale_layout": CUTLASS_SCALE_LAYOUT,
            "scale_dtype": "fp8-e4m3fn",
            "tensor_scale_dtype": FORMAT_F32,
            "input_scale_dtype": FORMAT_F32,
        },
        "expert_record": {"alignment": 4096, "immutable": True, "independently_addressable": True},
        "cutlass": {"version": "4.5.1", "revision": CUTLASS_REVISION},
    })
