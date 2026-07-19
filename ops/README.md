# Persistent service

The files in `ops/systemd/` run one persistent Colibri engine and expose the
same client-facing model names as the local DS4 service:

- `deepseek-v4-flash` and `deepseek-v4-pro` are advertised;
- `deepseek-chat` is accepted as a hidden non-thinking alias;
- thinking is enabled by default and can be disabled with `think: false`,
  `thinking: {"type": "disabled"}`, or the hidden alias.

Install the user service:

```bash
mkdir -p ~/.config/colibri ~/.config/systemd/user
cp ops/service.env.example ~/.config/colibri/service.env
cp ops/systemd/colibri-*.service ops/systemd/colibri-watchdog.timer \
  ~/.config/systemd/user/
chmod +x ops/colibri-watchdog.sh
chmod 600 ~/.config/colibri/service.env
systemctl --user daemon-reload
systemctl --user enable --now colibri-server.service colibri-watchdog.timer
loginctl enable-linger "$USER"
```

Two example env files ship here: `service.env.example` (unified-memory /
Spark-class hosts) and `service.env.tekton.example` (discrete multi-GPU,
"profile A").  **Start from the one that matches the machine** — they differ in
settings that silently cost an order of magnitude, notably `AUTOPIN`, which must
NOT be disabled where a VRAM expert tier is used.  The hardware-tuning skill
documents the per-profile reasoning and the measured service gotchas.

`ExecStartPre` runs `ops/colibri-preflight.sh`, which refuses to start when the
snapshot is missing or short (tmpfs does not survive a reboot), when
`nvidia_uvm` lacks `uvm_disable_hmm=1`, or when `python3` predates 3.7 — each of
which otherwise fails minutes into startup, or invisibly if the service user
cannot read the journal.  Set `COLI_PREFLIGHT_*` in the env file to enable the
checks that apply to the host.

Edit `service.env` first. `COLI_CONTEXT` is both the allocated engine context
and the value advertised to clients. KV memory grows with context and slots, so
increase `COLI_KV_SLOTS` only after measuring available memory.

The watchdog waits through startup, requires an idle GPU, and then sends a
one-token non-thinking inference request. It restarts only when that request
times out while the GPU remains idle. Before restart it records service state,
GPU state, sockets, journal output, and a best-effort gdb backtrace under
`~/.colibri/wedge-diag-*.log`.

To exercise the recovery path without restarting:

```bash
COLI_WATCHDOG_DRY_RUN=1 COLI_WATCHDOG_DIAGNOSTICS=0 \
COLI_WATCHDOG_URL=http://127.0.0.1:1 ops/colibri-watchdog.sh
```
