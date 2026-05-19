#!/bin/bash
# Wait for a specific checkpoint's HF save to FULLY complete, then SIGTERM the
# running train_lingbotvla.py processes for a clean stop.
#
# Completion marker: lingbotvla_cli.yaml inside the hf_ckpt dir. That file is
# written by save_args() AFTER every safetensors shard, config.json, and the
# tokenizer files are on disk, so its presence guarantees the save finished.
# Watching the hf_ckpt dir itself is wrong — the dir is created at the START
# of the save and waiting on it catches mid-write (truncated shards).
#
# Args:
#   $1 — abs path to the hf_ckpt dir (the script waits for $1/lingbotvla_cli.yaml)
# Env (optional):
#   POST_DELAY (default 15) — seconds to sleep after marker appears, before SIGTERM
#   POLL      (default 30) — seconds between checks
set -u

TARGET=${1:?usage: kill_after_step.sh <abs-path-to-hf_ckpt-dir>}
MARKER="$TARGET/lingbotvla_cli.yaml"
POST_DELAY=${POST_DELAY:-15}
POLL=${POLL:-30}
LOG=/home/azureuser/lingbot-vla/kill_after_step.log

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" >> "$LOG"; }

log "killer started (pid=$$). Waiting for $MARKER (poll=${POLL}s, post_delay=${POST_DELAY}s)."

while [ ! -f "$MARKER" ]; do
  if ! pgrep -f 'train_lingbotvla.py' >/dev/null; then
    log "training process is already gone — nothing to do, exiting."
    exit 0
  fi
  sleep "$POLL"
done
log "completion marker appeared: $MARKER"
sleep "$POST_DELAY"
log "sending SIGTERM to train_lingbotvla.py processes."
pkill -TERM -f 'train_lingbotvla.py'
sleep 5
# Best-effort follow up if anything is still around
pkill -TERM -f 'train_lingbotvla.py' 2>/dev/null || true
log "SIGTERM dispatched; orchestrator will pick up when processes exit."
