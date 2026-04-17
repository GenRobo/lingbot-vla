# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LingBot-VLA is a Vision-Language-Action (VLA) foundation model for dual-arm robotics. The primary workflow in this repo is **fine-tuning the pretrained model on a custom downstream task dataset**. Use `/finetune` to walk through the full pipeline interactively.

## Environment

- Python 3.12.3, PyTorch 2.8.0, CUDA 12.8
- `transformers==4.51.3`, `numpy==1.26.4`, `torchcodec==0.6.0`
- LeRobot pinned to commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`
- Flash Attention 2.8.3 (local `.whl` install)

## Fine-Tuning Pipeline (Custom Dataset)

This is the end-to-end workflow for adapting LingBot-VLA to a new task. Use `/finetune` to run this interactively.

### Step 1 — Data format

Data must be in **LeRobot format** (HuggingFace `datasets` with a `meta/` folder). The required keys depend on which dataset class you use, but for custom data (`CustomizedRobotwinDataset`, `data_type='customized'`) the expected keys are:
- `action` — `(chunk, action_dim)` tensor
- `observation.state` — `(state_dim,)` tensor
- `observation.images.cam_high`, `observation.images.cam_left_wrist`, `observation.images.cam_right_wrist` — uint8 RGB images
- `task` — string language instruction per episode

### Step 2 — Compute normalization stats

Run once per dataset. Outputs a JSON file used by `Normalizer` at training and deploy time.

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_robotwin_5.py configs/norm/robotwin_5.yaml \
  --model.model_path /path/to/lingbot-vla-4b \
  --model.tokenizer_path /path/to/Qwen2.5-VL-3B-Instruct \
  --data.train_path /path/to/your_lerobot_dataset \
  --data.norm_path assets/norm_stats/your_dataset.json
```

The script (`scripts/compute_norm_robotwin_5.py`) iterates the dataset and computes per-key `mean`, `std`, `q01`, `q99`. It expects keys `observation.state` and `action` — if your dataset uses different keys, edit lines 73–75 of that script.

### Step 3 — Register dataset class

For custom data, `CustomizedRobotwinDataset` in `lingbotvla/data/vla_data/base_dataset.py` is the template. It uses `data_type='customized'` in `Normalizer`, which simply converts all stats entries to numpy arrays without key remapping. If you need different key names or state concatenation logic, subclass from there.

Register the class in `lingbotvla/data/vla_data/__init__.py`:
```python
from .base_dataset import liberoDataset, RobotwinDataset, CustomizedRobotwinDataset, BM100Dataset, YourDataset
```

Then add a branch in `tasks/vla/train_lingbotvla.py` around line 330:
```python
elif args.data.data_name == 'your_dataset_name':
    train_dataset = YourDataset(repo_id=args.data.train_path, ...)
```

### Step 4 — Create a config YAML

Copy `configs/vla/robotwin_load20000h.yaml` as a starting point. Key fields to change:

```yaml
model:
  model_path: /path/to/lingbot-vla-4b        # pretrained checkpoint
  tokenizer_path: /path/to/Qwen2.5-VL-3B-Instruct
  post_training: true
  adanorm_time: true     # must be true for all LingBot-VLA checkpoints
  old_adanorm: true      # must be true for all LingBot-VLA checkpoints

data:
  data_name: your_dataset_name               # matches branch added in step 3
  train_path: /path/to/your_lerobot_dataset
  norm_stats_file: assets/norm_stats/your_dataset.json
  norm_type: bounds_99_woclip                # recommended default

train:
  output_dir: /path/to/output
  action_dim: 14          # actual DOF of your robot
  max_action_dim: 75      # keep at 75 (model's internal padding size)
  max_state_dim: 75       # keep at 75
  loss_type: L1_fm        # recommended for post-training
  micro_batch_size: 8     # per-GPU batch size
  global_batch_size: 32   # micro_batch_size × num_GPUs
  num_train_epochs: 69
  save_steps: 10000
  save_hf_weights: true   # extract HF-format checkpoint alongside DCP
  enable_resume: true
```

### Step 5 — Run training

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/your_config.yaml \
  --model.model_path /path/to/lingbot-vla-4b \
  --model.tokenizer_path /path/to/Qwen2.5-VL-3B-Instruct \
  --data.train_path /path/to/your_dataset \
  --train.output_dir /path/to/output \
  --train.micro_batch_size 8 \
  --train.global_batch_size 32
```

`train.sh` auto-detects GPU count via `nvidia-smi` or `CUDA_VISIBLE_DEVICES`. Multi-node: set `NNODES`, `NODE_RANK`, `MASTER_ADDR`. Logs tee'd to `log.txt`. TensorBoard under `{output_dir}/runs/`. Per-step loss in `{output_dir}/checkpoints/loss.jsonl`.

Checkpoints saved under `{output_dir}/checkpoints/global_step_{N}/`. With `save_hf_weights: true`, HF weights extracted to `hf_ckpt/` inside the checkpoint dir. Resume is automatic — training scans for the latest `global_step_*` dir on restart.

### Step 6 — Deploy

The deploy server must use **identical key mapping and `data_type`** as the dataset class used during training. Use `lingbot_robotwin_policy.py` as a reference and replicate the normalizer construction.

```bash
export QWEN25_PATH=/path/to/Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_robotwin_policy \
  --model_path /path/to/output/checkpoints/global_step_N/hf_ckpt \
  --use_length 50 \
  --port 8765
```

For open-loop evaluation (no robot, compare predicted vs GT actions):
```bash
python scripts/open_loop_eval.py
```

## Normalization `data_type` Reference

`Normalizer` in `lingbotvla/data/vla_data/transform.py` applies different key remapping per `data_type`:

| `data_type` | Behavior |
|---|---|
| `customized` | No remapping — loads stats keys as-is (use this for new datasets) |
| `robotwin` | Remaps split arm/effector keys into `observation.state` and `action` |
| `robotwin_rep` | Same as `robotwin` but without the 6+1 interleaving |
| `bm100` | Leaves top-level keys, just converts to numpy |
| `libero` | Truncates state to 8D and actions to 7D |

## Critical Constraints

- `adanorm_time: true` and `old_adanorm: true` must both be set when loading any released LingBot-VLA checkpoint — mismatches throw an assertion at model build time.
- `data_type` in `Normalizer` must be identical between training (`base_dataset.py`) and deployment (`deploy/*.py`), otherwise normalization is applied incorrectly.
- `RoboTwin/` is a git submodule — do not edit files inside it.
- Do not commit anything from `ckpts/`, `data/`, `output/`, `results/`.

## Branch Strategy

```
main                     ← read-only, tracks upstream robbyant/lingbot-vla only
  └── gr/main            ← our stable base; merge main → here when pulling upstream updates
        └── feature/*    ← all development branches; PR into gr/main when ready
```

**Remotes:**
- `origin` → our fork (e.g. `generalrobotics/lingbot-vla`)
- `upstream` → `https://github.com/Robbyant/lingbot-vla.git`

**Pulling upstream updates into our codebase:**
```bash
git fetch upstream
git checkout main && git merge upstream/main && git push origin main
git checkout gr/main && git merge main && git push origin gr/main
```

**Never commit custom work directly to `main`.** It exists solely to mirror upstream so merges stay clean.

## Other Commands

```bash
# Download pretrained weights
python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-4b --local_dir ckpts/lingbot-vla-4b

# Lint
ruff check lingbotvla/
ruff format lingbotvla/

# Tests
pytest
```
