#!/bin/bash
# Run open-loop eval on a set of checkpoints, aggregate results, plot the curve.
# Args:
#   $1 — output dir to scan for checkpoints (e.g. output/r1pro_delta_right)
#   $2 — eval dir to write results into (e.g. eval/r1pro_delta_right)
# Env (optional):
#   STEPS         — space-separated list of global_step numbers to score (defaults to every-4th saved ckpt)
#   TRAJ_IDS      — space-separated trajectory indices (default: 0 13 26 39 52 65 78 91 104 117)
#   USE_LENGTH    — action chunk length used at inference (default 25)
#   NUM_DENOISE   — denoising steps (default 10)
#   GPU           — CUDA device to pin (default 0)

set -euo pipefail

OUT_ROOT=${1:?usage: run_open_loop_eval.sh <train-output-dir> <eval-dir>}
EVAL_ROOT=${2:?usage: run_open_loop_eval.sh <train-output-dir> <eval-dir>}
REPO=/home/azureuser/lingbot-vla
PY=/home/azureuser/micromamba/envs/lingbotvla/bin/python
LOG=$EVAL_ROOT/run.log
DATA_PATH=/home/azureuser/dataset/htx/brs_ctrl/lerobot_v3/iv_pnp_250515

TRAJ_IDS=${TRAJ_IDS:-"0 13 26 39 52 65 78 91 104 117"}
USE_LENGTH=${USE_LENGTH:-25}
NUM_DENOISE=${NUM_DENOISE:-10}
GPU=${GPU:-0}

mkdir -p "$EVAL_ROOT"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# Pick steps if not provided: every 4th saved checkpoint, plus the last.
if [ -z "${STEPS:-}" ]; then
  mapfile -t all_steps < <(
    find "$OUT_ROOT/checkpoints" -maxdepth 1 -type d -name 'global_step_*' \
      | awk -F'_' '{print $NF}' | sort -n
  )
  if [ ${#all_steps[@]} -eq 0 ]; then
    log "ERROR: no checkpoints found under $OUT_ROOT/checkpoints"
    exit 1
  fi
  picked=()
  for i in $(seq 0 4 $((${#all_steps[@]}-1))); do
    picked+=("${all_steps[$i]}")
  done
  # always include the last one
  last="${all_steps[$((${#all_steps[@]}-1))]}"
  if [[ ! " ${picked[*]} " =~ " ${last} " ]]; then
    picked+=("$last")
  fi
  STEPS="${picked[*]}"
fi

log "OUT_ROOT=$OUT_ROOT EVAL_ROOT=$EVAL_ROOT GPU=$GPU"
log "checkpoint steps: $STEPS"
log "traj_ids: $TRAJ_IDS  use_length=$USE_LENGTH  num_denoising_step=$NUM_DENOISE"

export QWEN25_PATH=Qwen/Qwen2.5-VL-3B-Instruct
export CUDA_VISIBLE_DEVICES=$GPU

cd "$REPO"

for step in $STEPS; do
  CKPT="$OUT_ROOT/checkpoints/global_step_${step}/hf_ckpt"
  if [ ! -d "$CKPT" ]; then
    log "SKIP step_${step}: missing $CKPT (likely pruned without being evaluated; or not yet saved)"
    continue
  fi
  SAVE="$EVAL_ROOT/step_${step}"
  mkdir -p "$SAVE"
  log "--- evaluating step_${step} ---"
  $PY scripts/open_loop_eval.py \
      --model_path "$CKPT" \
      --data_path "$DATA_PATH" \
      --use_length "$USE_LENGTH" \
      --num_denoising_step "$NUM_DENOISE" \
      --use_compile \
      --traj_ids $TRAJ_IDS \
      --save_plot_path "$SAVE" \
      2>&1 | tee "$SAVE/stdout.log" \
      || log "FAILED step_${step} (continuing)"
done

# Aggregate per-step MSE/MAE from each stdout.log and plot the curve.
log "aggregating summary"
$PY - "$EVAL_ROOT" <<'PYEOF'
import json, os, re, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for d in sorted(root.glob("step_*")):
    step = int(d.name.split("_")[1])
    log_path = d / "stdout.log"
    if not log_path.exists():
        continue
    text = log_path.read_text()
    per_traj = re.findall(r"MSE for trajectory (\d+): ([0-9.eE+-]+), MAE: ([0-9.eE+-]+)", text)
    avg_mse = re.search(r"Average MSE across all trajs: ([0-9.eE+-]+)", text)
    avg_mae = re.search(r"Average MAE across all trajs: ([0-9.eE+-]+)", text)
    summary[step] = {
        "mse_avg": float(avg_mse.group(1)) if avg_mse else None,
        "mae_avg": float(avg_mae.group(1)) if avg_mae else None,
        "per_traj": [{"traj_id": int(t), "mse": float(m), "mae": float(a)} for t, m, a in per_traj],
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"summary -> {root}/summary.json")
print(json.dumps({k: {"mse_avg": v["mse_avg"], "mae_avg": v["mae_avg"]} for k, v in summary.items()}, indent=2))

# Plot curve
try:
    import matplotlib.pyplot as plt
    steps = sorted(k for k, v in summary.items() if v["mse_avg"] is not None)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [summary[s]["mse_avg"] for s in steps], "o-")
    ax[0].set_xlabel("global step"); ax[0].set_ylabel("avg unnormalized action MSE"); ax[0].grid(True)
    ax[1].plot(steps, [summary[s]["mae_avg"] for s in steps], "o-", color="orange")
    ax[1].set_xlabel("global step"); ax[1].set_ylabel("avg unnormalized action MAE"); ax[1].grid(True)
    fig.suptitle(f"Open-loop eval on training data ({root.name})")
    fig.tight_layout()
    fig.savefig(root / "curve.png", dpi=110)
    print(f"curve -> {root}/curve.png")
except Exception as e:
    print(f"plot failed: {e}")
PYEOF

log "eval complete: $EVAL_ROOT/summary.json + curve.png"
