"""TAP/INJECT validation against a transformers oracle.

Procedure (all CPU, ~2 min):
  1. tools/make_glm_oracle.py            -> glm_tiny/ + ref_glm.json
  2. tools/convert_fp8_to_int4.py --indir glm_tiny --outdir glm_tiny_i8 \
         --ebits 8 --io-bits 8
  3. make glm
  4. python3 tests/test_tap.py           (run from c/)

Checks:
  - TAP at layer L matches transformers hidden_states[L+1] (mean cos > 0.98
    at int8; the residue is container quantization).
  - TAP does not perturb generation (byte-identical tokens).
  - INJECT of a zero vector is a no-op (byte-identical tokens).
  - INJECT of a unit vector at scale 3 changes generation.

IMPORTANT: the reference model MUST be loaded with attn_implementation="eager".
The sdpa path of GlmMoeDsa diverges from eager (measured: TF self-match 0/12
on the tiny fixture) — the engine is faithful to eager, which is also what
generated ref_glm.json.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import GlmMoeDsaForCausalLM

D = 128  # hidden of the tiny fixture
SNAP = "./glm_tiny_i8"


def run_engine(env_extra):
    env = dict(os.environ, SNAP=SNAP, **env_extra)
    out = subprocess.run(["./glm"], env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"engine exited {out.returncode}:\n{out.stdout}\n{out.stderr}")
    for line in out.stdout.splitlines():
        if line.startswith("Motore C GLM"):
            return line
    sys.exit(f"engine produced no generation line:\n{out.stdout}\n{out.stderr}")


def main():
    ref = json.load(open("ref_glm.json"))
    prompt = ref["prompt_ids"]

    base = run_engine({})

    # --- TAP fidelity + non-perturbation, every layer ---
    model = GlmMoeDsaForCausalLM.from_pretrained(
        "glm_tiny", attn_implementation="eager").eval()
    with torch.no_grad():
        hs = model(torch.tensor([prompt]), output_hidden_states=True,
                   use_cache=False).hidden_states
    n_layers = len(hs) - 1
    for L in range(n_layers):
        tapf = f"/tmp/coli_tap_l{L}.bin"
        line = run_engine({"TAP": f"{L}:{tapf}"})
        assert line == base, f"TAP perturbed generation at L{L}"
        recs = np.fromfile(tapf, dtype=np.float32).reshape(-1, 1 + D)
        v = recs[recs[:, 0].astype(int) < len(prompt)][:, 1:]
        h = hs[L + 1][0].float().numpy()
        cs = [float(np.dot(v[p], h[p]) /
                    (np.linalg.norm(v[p]) * np.linalg.norm(h[p])))
              for p in range(len(prompt))]
        m = float(np.mean(cs))
        print(f"TAP L{L} vs eager hidden_states[{L+1}]: mean cos {m:.4f}")
        assert m > 0.98, f"TAP L{L} diverges from oracle (cos {m:.4f})"

    # --- INJECT: zero no-op, nonzero changes output ---
    np.zeros(D, dtype=np.float32).tofile("/tmp/coli_inj_zero.bin")
    rng = np.random.RandomState(7)
    v = rng.randn(D).astype(np.float32)
    (v / np.linalg.norm(v)).tofile("/tmp/coli_inj_rand.bin")

    assert run_engine({"INJECT": "3:/tmp/coli_inj_zero.bin"}) == base, \
        "zero INJECT is not a no-op"
    print("INJECT zero vector: no-op OK")
    assert run_engine({"INJECT": "3:/tmp/coli_inj_rand.bin",
                       "INJECT_SCALE": "3.0"}) != base, \
        "INJECT scale 3.0 did not change generation"
    print("INJECT unit*3.0: changes generation OK")

    # --- hook-order oracle: at the SAME layer TAP runs before INJECT
    # (sensor reads the un-steered state), one layer up it must see it ---
    def prefill_rows(tapf):
        recs = np.fromfile(tapf, dtype=np.float32).reshape(-1, 1 + D)
        return recs[recs[:, 0].astype(int) < len(prompt)][:, 1:]

    run_engine({"TAP": "3:/tmp/coli_tap_same.bin",
                "INJECT": "3:/tmp/coli_inj_rand.bin", "INJECT_SCALE": "3.0"})
    clean = prefill_rows("/tmp/coli_tap_l3.bin")       # from the TAP-only pass
    same = prefill_rows("/tmp/coli_tap_same.bin")
    assert np.array_equal(clean, same), \
        "TAP at the inject layer must record the PRE-inject residual"
    print("TAP@L3 + INJECT@L3: tap reads pre-inject state OK")

    run_engine({"TAP": "4:/tmp/coli_tap_down.bin",
                "INJECT": "3:/tmp/coli_inj_rand.bin", "INJECT_SCALE": "3.0"})
    down = prefill_rows("/tmp/coli_tap_down.bin")
    clean4 = prefill_rows("/tmp/coli_tap_l4.bin")
    assert not np.array_equal(clean4, down), \
        "TAP one layer above the inject must see the steering"
    print("TAP@L4 + INJECT@L3: steering visible downstream OK")
    print("ALL TAP/INJECT CHECKS PASSED")


if __name__ == "__main__":
    main()
