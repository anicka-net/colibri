#!/usr/bin/env python3
"""Generate the fmt=6 kernel fixture from the reference codec (#452).

tests/test_e8_kernel.c checks the C kernel against these bytes, so the fixture
must come from iq3_pack.py — the format's reference implementation — and never
from the kernel itself. Regenerate whenever the layout changes:

    python3 tools/make_e8_fixture.py
"""
import os
import struct

import numpy as np

import iq3_pack as P

O, I = 24, 512
SEED = 20260721


def main():
    rng = np.random.default_rng(SEED)
    w = (rng.standard_normal((O, I)) * 0.05).astype(np.float32)
    packed = P.encode(w)
    deq = P.decode(packed, I).astype(np.float32)
    x = (rng.standard_normal(I) * 1.0).astype(np.float32)
    # float64 reference so the C float path is compared against something better
    yref = (deq.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)

    out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "e8_case.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", O, I))
        f.write(packed.tobytes())
        f.write(deq.tobytes())
        f.write(x.tobytes())
        f.write(yref.tobytes())
    print(f"wrote {path}: O={O} I={I} packed={packed.nbytes}B "
          f"({packed.nbytes * 8 / (O * I):.4f} bpw)")


if __name__ == "__main__":
    main()
