#!/bin/bash
# Watch output dirs and prune everything except hf_ckpt/ from non-latest checkpoint dirs.
# Latest global_step_N keeps full DCP state (for resume). All older keep only hf_ckpt/.
# Run in background:   bash scripts/checkpoint_janitor.sh &
set -u

OUT_DIRS=(
  /home/azureuser/lingbot-vla/output/r1pro_delta_right
  /home/azureuser/lingbot-vla/output/r1pro_delta_dual
)
LOG=/home/azureuser/lingbot-vla/checkpoint_janitor.log
INTERVAL=${INTERVAL:-30}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" >> "$LOG"; }

log "janitor started (pid=$$ interval=${INTERVAL}s dirs=${OUT_DIRS[*]})"

while true; do
  for OUT in "${OUT_DIRS[@]}"; do
    CKPT="$OUT/checkpoints"
    [ -d "$CKPT" ] || continue
    # Only consider "completed" checkpoints — those that have hf_ckpt/ written
    # (saved AFTER the DCP state). This guards against pruning while a newer
    # save is still in progress.
    mapfile -t completed < <(
      find "$CKPT" -maxdepth 1 -type d -name 'global_step_*' \
        | while read d; do [ -d "$d/hf_ckpt" ] && echo "$d"; done \
        | awk -F'_' '{print $NF"\t"$0}' | sort -k1 -n -r | cut -f2
    )
    n=0
    for d in "${completed[@]}"; do
      n=$((n+1))
      [ $n -eq 1 ] && continue
      # 2nd-and-older completed dirs: drop everything except hf_ckpt/
      for sub in "$d"/*; do
        bn=$(basename "$sub")
        if [ "$bn" != "hf_ckpt" ] && [ -e "$sub" ]; then
          sz=$(du -sb "$sub" 2>/dev/null | cut -f1)
          rm -rf "$sub" && log "pruned $sub ($(numfmt --to=iec "${sz:-0}"))"
        fi
      done
    done
  done
  sleep "$INTERVAL"
done
