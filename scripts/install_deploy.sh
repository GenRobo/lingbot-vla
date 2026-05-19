#!/usr/bin/env bash
# Create the `lingbotvla-deploy` env for running the LingbotVLA inference
# server and the R1Pro robot driver. Separate from the training env
# (`lingbotvla`) so deployment can be installed/upgraded independently.
#
# What goes in:
#   - Python 3.12 + PyTorch 2.8 (CUDA 12.8)
#   - lerobot 0.4.2, transformers 4.51.3, lingbotvla (editable)
#   - flash-attn 2.8.3 — required: the Qwen backbone loads with
#     `_attn_implementation=flash_attention_2` regardless of the server's
#     `attention_implementation='eager'` override (different field name).
#   - GRID-Robot-API (editable) for the driver — uses whatever branch you
#     have checked out in $GRA_DIR
#
# Usage:
#     bash scripts/install_deploy.sh                # standard install
#     ENV_NAME=foo bash scripts/install_deploy.sh   # different env name
#     CLONE_FROM=lingbotvla bash scripts/install_deploy.sh
#         # Faster path for people who already have the training env on the
#         # same machine: clone it instead of re-downloading everything.
#         # Note: micromamba clone only copies conda packages, so we still
#         # re-run pip on top to fill in the gaps.
#
# Prereqs:
#   - micromamba (or mamba/conda) on PATH
#   - $GRA_DIR points to a GRID-Robot-API checkout (default: sibling of
#     this repo). Make sure GRA is on the branch you want before running.
#   - For driver execution only: ROS2 installed system-wide; `source
#     /opt/ros/<distro>/setup.bash` after activating the env, so rclpy
#     resolves at import time.

set -euo pipefail

ENV_NAME="${ENV_NAME:-lingbotvla-deploy}"
CLONE_FROM="${CLONE_FROM:-}"
PY_VERSION="${PY_VERSION:-3.12}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRA_DIR="${GRA_DIR:-$(cd "$REPO_DIR/.." && pwd)/GRID-Robot-API}"

# Pick an env manager. micromamba first (matches the existing setup on
# our internal laptops); fall back to mamba then conda.
if command -v micromamba >/dev/null 2>&1; then
    MGR="micromamba"
elif command -v mamba >/dev/null 2>&1; then
    MGR="mamba"
elif command -v conda >/dev/null 2>&1; then
    MGR="conda"
else
    echo "error: none of micromamba/mamba/conda found on PATH" >&2
    exit 1
fi
echo "==> Using env manager: $MGR"
RUN="$MGR run -n $ENV_NAME"

env_exists() {
    $MGR env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

# --- create env ---------------------------------------------------------
if env_exists "$ENV_NAME"; then
    echo "==> Env '${ENV_NAME}' exists — installing on top of it."
elif [ -n "$CLONE_FROM" ] && env_exists "$CLONE_FROM"; then
    echo "==> Cloning '${CLONE_FROM}' -> '${ENV_NAME}' (fast path)"
    $MGR create -n "$ENV_NAME" --clone "$CLONE_FROM" -y
else
    if [ -n "$CLONE_FROM" ]; then
        echo "warning: CLONE_FROM='${CLONE_FROM}' not found — doing fresh install instead." >&2
    fi
    echo "==> Creating fresh env '${ENV_NAME}' (python=${PY_VERSION})"
    $MGR create -n "$ENV_NAME" "python=${PY_VERSION}" -y -c conda-forge
fi

$RUN pip install --upgrade pip

# --- install deps (idempotent on top of either fresh or cloned env) ---
echo "==> Installing PyTorch (CUDA 12.8)"
$RUN pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing lerobot 0.4.2 (no-deps; we pin its conflicting deps via requirements.txt)"
$RUN pip install lerobot==0.4.2 --no-deps

echo "==> Installing lingbot-vla deps (pinned, overrides lerobot's looser pins)"
$RUN pip install -r "$REPO_DIR/requirements.txt"

# lerobot needs a few packages that aren't in our requirements.txt.
# Explicit list rather than `lerobot[everything]` to keep the env lean.
echo "==> Installing lerobot's runtime deps we still need"
$RUN pip install accelerate draccus blobfile av einops huggingface-hub safetensors torchdata "torchcodec==0.6.0"

echo "==> Installing flash-attn 2.8.3 (~5 min compile — required by Qwen backbone)"
$RUN pip install flash-attn==2.8.3 --no-build-isolation

echo "==> Installing lingbot-vla (editable)"
$RUN pip install -e "$REPO_DIR" --no-deps

# Driver-side extras. tyro for CLI, opencv-python in case GRA's JPEG
# decode path isn't used.
echo "==> Installing driver extras (tyro, opencv-python)"
$RUN pip install tyro opencv-python

# GRA — pip-editable so switching branches doesn't require a reinstall.
if [ -d "$GRA_DIR" ]; then
    GRA_BRANCH="$(git -C "$GRA_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")"
    echo "==> Installing GRID-Robot-API (editable) from $GRA_DIR (branch: $GRA_BRANCH)"
    $RUN pip install -e "$GRA_DIR"
else
    cat >&2 <<EOF
warning: GRID-Robot-API not found at $GRA_DIR — skipping GRA install.
The inference server will still work, but the robot driver
(deploy/r1pro_real/main.py) will fail to import rclpy/GRA. To install
later:
    git clone https://github.com/GenRobo/GRID-Robot-API.git
    cd GRID-Robot-API && git checkout <branch>
    $MGR run -n $ENV_NAME pip install -e .
EOF
fi

# --- final sanity check -------------------------------------------------
echo "==> Sanity check"
$RUN python -c "
import torch, transformers, lerobot, lingbotvla, websockets, msgpack
print('  python deps    OK')
print('  cuda available:', torch.cuda.is_available())
try:
    import grid_robot_api
    print('  grid_robot_api OK at', grid_robot_api.__file__)
except ImportError:
    print('  grid_robot_api NOT installed (driver will not run; server-only OK)')
"

cat <<EOF

============================================================
Install complete.
============================================================

Quick start (run from repo root: $REPO_DIR):

  # 1. Start the inference server
  $MGR run -n $ENV_NAME --cwd $REPO_DIR python -m deploy.lingbot_vla_policy \\
      --model_path output/r1pro_delta_right/checkpoints/global_step_4760/hf_ckpt \\
      --port 8000 --use_length 25 --use_compile

  # 2. (Optional) Verify with the synthetic dry-run before the robot
  $MGR run -n $ENV_NAME --cwd $REPO_DIR python -m deploy.r1pro_real.dry_run \\
      --remote_host=localhost --remote_port=8000

  # 3. Source ROS2 in the same shell as the driver
  source /opt/ros/<distro>/setup.bash
  $MGR run -n $ENV_NAME --cwd $REPO_DIR python -m deploy.r1pro_real.main \\
      --remote_host=localhost --remote_port=8000 \\
      --instruction "pick up the apple"

See deploy/README.md for full details.
EOF
