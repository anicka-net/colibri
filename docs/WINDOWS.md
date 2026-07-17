# Windows 11 native install — a complete walkthrough (no WSL)

A start-to-finish, reproducible path from a fresh Windows 11 machine to GLM-5.2 generating tokens, with the GPU tier. Every step and every failure mode below was hit and verified on real hardware: Core Ultra 9 285K (AVX-VNNI) / RTX 5080 (sm_120) / 128 GB RAM / Windows 11 24H2 (issue #306). Steps are ordered so the long downloads run while you build.

## 0. What you need

| Piece | Why | Get it |
|---|---|---|
| git, Python 3 | clone + `coli` launcher | winget / python.org |
| MinGW-w64 gcc + make | builds the engine (MSVC can't) | `scoop install mingw-winlibs`, MSYS2, or portable **w64devkit** (no admin, unzip and go) |
| CUDA Toolkit ≥ 12.8 | GPU tier; ≥12.8 required for Blackwell/sm_120 | `winget install Nvidia.CUDA` |
| MSVC Build Tools (C++ workload) | nvcc's host compiler for the CUDA DLL | `winget install Microsoft.VisualStudio.2022.BuildTools` + "Desktop development with C++" |
| ~400 GB free on a local NVMe | the int4 model (~370–384 GB) | NTFS is fine; **never** a network mount |

RAM: 16 GB minimum, more = bigger expert cache = faster. The build itself needs none of the CUDA/MSVC pieces — do the CPU build first, add the GPU tier later.

## 1. Start the model download first (it's the long pole)

```powershell
python -m pip install -U "huggingface_hub[hf_transfer]"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
hf download <model-repo> --local-dir D:\glm52_i4
```

Use the container recommended in the README (with **int8 MTP heads** — int4 heads silently give 0% draft acceptance). The download is resumable: if it stops, rerun the same command. Expect hours; everything below fits inside them.

## 2. Build the engine (CPU)

From a normal PowerShell, in the repo's `c\` directory:

```powershell
make glm.exe ARCH=native      # ARCH=native unlocks AVX-VNNI on Alder Lake+/Arrow Lake
make iobench.exe              # disk benchmark, useful before committing to the download
```

Warnings about `#pragma comment` and unused variables are normal (MSVC-isms gcc ignores). The engine banner should print `idot: avx-vnni` on VNNI-capable CPUs — if it says avx2, you built without `ARCH=native`.

### ⚠️ Smart App Control will block your fresh binary

On Windows 11 machines with **Smart App Control** enforced (`VerifiedAndReputablePolicyState = 1`), running your self-compiled `glm.exe` fails with:

```
Program 'glm.exe' failed to run: An Application Control policy has blocked this file
```

This is not Defender and not Mark-of-the-Web — SAC blocks *all* unsigned, unknown binaries, which includes anything you compile yourself. **Fix:** Windows Security → App & browser control → Smart App Control settings → **Off**, then **reboot** (the policy only reloads on restart). Note SAC is one-way: re-enabling later requires resetting Windows. If the settings page is missing, the registry equivalent is setting `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState` to `0` (admin PowerShell), then rebooting. Check your current state before touching anything:

```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy").VerifiedAndReputablePolicyState
# 0 = off, 1 = enforced, 2 = evaluation
```

## 3. Build the CUDA DLL (GPU tier)

nvcc needs MSVC as host compiler, so this one step must run from a shell with the MSVC environment: open **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu (plain PowerShell will fail the `cl` check). Then:

```cmd
make cuda-dll CUDA_ARCH=sm_120        # match your GPU: sm_120 Blackwell, sm_89 Ada, ...
make glm.exe CUDA_DLL=1 ARCH=native   # relink host with the runtime loader
```

Two pitfalls, both fixed on current `dev` (#314) but worth knowing on older checkouts:

- **Spaces in `CUDA_HOME`** (`C:\Program Files\...`) used to break the recipe → fixed; nvcc now comes from PATH and `"$(NVCC)"` is quoted.
- **`make glm.exe CUDA_DLL=1` after a CPU-only build** used to report `up to date` and silently keep the CPU-only binary (GPU tier never engages, no error). Current `dev` has a build-config stamp that forces the relink. On older trees: delete `glm.exe` first.

Sanity check: first GPU run should print `[CUDA] device 0: <your GPU>, ... sm_XX` and `[CUDA] mode: routed experts + resident dense tensors`.

## 4. First run

```powershell
cd <repo>\c
$env:OMP_NUM_THREADS = "<physical cores>"
python coli run "Explain what a mixture-of-experts model is." --model D:\glm52_i4 --ngen 48
```

The first run is cold — expect the profile to be dominated by `expert-disk` while the cache warms; hit rate climbs run over run. GPU tier on top:

```powershell
$env:COLI_CUDA="1"; $env:COLI_GPU="0"; $env:CUDA_DENSE="1"; $env:CUDA_EXPERT_GB="4"
python coli run "..." --model D:\glm52_i4 --ngen 64
```

Size `CUDA_EXPERT_GB` so dense (~10 GB) + experts + working set stays under your VRAM. Note MTP speculation is off by default under CUDA (#293, float-accumulation divergence between draft and verify) — `COLI_CUDA_MTP=1` opts back in.

## 5. Reference numbers from this walkthrough's hardware

285K / RTX 5080 / 128 GB / NVMe at 5.85 GB/s random-read (19 MB blocks, `iobench`): 0.26 tok/s cold CPU → 0.30 warm CPU (MTP 2.2–2.3 tok/forward) → 0.42 tok/s GPU tier + auto-pin, expert hit 66%, ~65% of wall time in expert-disk. Disk-bound is the expected shape at ~25% expert residency — a faster disk and more RAM move the floor, the GPU moves the compute.

## Quick failure index

| Symptom | Cause | Fix |
|---|---|---|
| `An Application Control policy has blocked this file` | Smart App Control | §2 — turn SAC off + **reboot** |
| `cuda-dll ... Error 1` immediately | old tree: spaced CUDA_HOME / MSVC rejects `-Wextra` | update to current `dev` (#314) |
| `glm.exe is up to date` but GPU never engages | old tree: stale CPU-only binary | update to `dev`, or delete `glm.exe` and rebuild |
| `cl.exe (MSVC) not in PATH` | built from plain PowerShell | use the x64 Native Tools prompt |
| `nvcc fatal: unsupported gpu architecture 'sm_120'` | CUDA < 12.8 | install CUDA 12.8+ |
| MTP `0% (0/0)` on CPU path | int4 MTP heads in the container | use the int8-MTP container |
| MTP `draft=0` under CUDA | intended default since #293 | `COLI_CUDA_MTP=1` to opt in |
