#!/bin/bash
# Startup guard for the Colibri service: refuse to start (loudly, and without
# burning a GPU-sized allocation) when the host is not actually ready.
#
# Two failure modes this exists for, both observed:
#   1. The snapshot lives on tmpfs, which is EMPTY after a reboot.  Starting
#      against a missing or half-copied model wastes a ~15 min load and, with
#      Restart=always, spins on the disk while the real copy is still running.
#   2. On the open NVIDIA kernel module, CUDA needs nvidia_uvm loaded with
#      uvm_disable_hmm=1.  Without it every CUDA call returns error 3 with no
#      diagnostics at all (open-gpu-kernel-modules #780/#797) — hours to
#      diagnose live, one line to catch here.
#
# All checks are opt-in through the service env file, so this is a no-op on
# hosts that do not need them.  Exit non-zero -> systemd retries per RestartSec.
set -u

fail() { echo "preflight: $*" >&2; exit 1; }

MODEL=${COLI_MODEL:-}
[ -n "$MODEL" ] || fail "COLI_MODEL is not set"
[ -d "$MODEL" ] || fail "model directory $MODEL does not exist (tmpfs not loaded after reboot?)"

for f in config.json tokenizer.json; do
    [ -r "$MODEL/$f" ] || fail "missing $MODEL/$f — snapshot incomplete"
done

min_shards=${COLI_PREFLIGHT_MIN_SHARDS:-1}
shards=$(find "$MODEL" -maxdepth 1 -name 'out-[0-9]*' -type f 2>/dev/null | wc -l)
[ "$shards" -ge "$min_shards" ] || \
    fail "found $shards weight shards, expected >= $min_shards — snapshot incomplete"

if [ "${COLI_PREFLIGHT_NEED_IDX:-0}" = "1" ]; then
    idx=$(find "$MODEL" -maxdepth 1 -name 'out-idx-*' -type f 2>/dev/null | wc -l)
    [ "$idx" -ge 1 ] || fail "no out-idx-* files — DSA sparse attention would be disabled"
fi

min_gb=${COLI_PREFLIGHT_MIN_GB:-0}
if [ "$min_gb" -gt 0 ]; then
    # du walks 165 files on tmpfs: milliseconds, and it catches the truncated
    # copies that a plain file-count check happily accepts.
    gb=$(du -sBG --apparent-size "$MODEL" 2>/dev/null | cut -f1 | tr -d 'G')
    [ -n "$gb" ] || fail "cannot size $MODEL"
    [ "$gb" -ge "$min_gb" ] || fail "snapshot is ${gb} GB, expected >= ${min_gb} GB — copy still running or truncated"
fi

# The HTTP server needs ThreadingHTTPServer (Python >= 3.7).  Distros whose
# default python3 is older (SLES 15's 3.6) fail 100 ms into ExecStart with an
# ImportError that never reaches the journal on a locked-down user session —
# set COLI_PYTHON to a newer interpreter and check it here.
PY=${COLI_PYTHON:-python3}
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,7) else 1)' 2>/dev/null; then
    have=$("$PY" -V 2>&1)
    fail "$PY is $have; the server needs >= 3.7 (ThreadingHTTPServer). Set COLI_PYTHON to a newer interpreter."
fi

if [ "${COLI_PREFLIGHT_UVM_HMM:-0}" = "1" ]; then
    p=/sys/module/nvidia_uvm/parameters/uvm_disable_hmm
    [ -r "$p" ] || fail "nvidia_uvm is not loaded (CUDA would fail with error 3)"
    [ "$(cat "$p")" = "Y" ] || fail "nvidia_uvm loaded WITHOUT uvm_disable_hmm=1 — every CUDA call will fail with error 3; fix: rmmod nvidia_uvm && modprobe nvidia_uvm uvm_disable_hmm=1 (persist via /etc/modprobe.d)"
fi

echo "preflight: ok — $shards shards in $MODEL"
