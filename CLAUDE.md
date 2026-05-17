# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LingBot-VLA is a Vision-Language-Action (VLA) foundation model for dual-arm robot control. The codebase focuses on post-training (fine-tuning) a pre-trained VLA checkpoint on custom robot datasets. The VLM backbone is Qwen2.5-VL-3B-Instruct; the action expert is a flow-matching diffusion policy.

**Environment**: Python 3.12, PyTorch 2.8, CUDA 12.8, conda env `lingbotvla`.

## Commands

```bash
ruff check . && ruff format .    # lint + format
```

### Training
```bash
# train.sh wraps torchrun and auto-detects GPU count
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/real_load20000h.yaml \
    --data.train_path /path/to/dataset \
    --data.data_name <robot_config_name> \
    --data.norm_stats_file assets/norm_stats/<name>.json \
    --train.output_dir output/

# Low-memory nodes (e.g. 4× A6000): reduce micro batch + increase accumulation
    --train.micro_batch_size 4 --train.gradient_accumulation_steps 16
```

### Compute Normalization Statistics
```bash
# Run before training whenever the dataset or robot config changes
python scripts/compute_norm.py configs/vla/<config>.yaml \
    --data.train_path /path/to/dataset \
    --data.data_name <robot_config_name>
```

### Evaluation
```bash
export QWEN25_PATH=Qwen/Qwen2.5-VL-3B-Instruct

# Open-loop eval
python scripts/open_loop_eval.py \
    --model_path output/checkpoints/global_step_XXXXX/hf_ckpt \
    --data_path /path/to/val_data --use_length 50

# Deploy inference server (real robot)
python -m deploy.lingbot_vla_policy \
    --model_path output/checkpoints/global_step_XXXXX/hf_ckpt \
    --use_compile --use_length 25
```

## Architecture

### Data Pipeline (must follow in order)

1. **LeRobot v3.0 dataset** — raw demonstrations in parquet + video format.

2. **Robot config YAML** (`configs/robot_configs/<name>.yaml`) — maps raw LeRobot feature keys to the unified feature space (`observation.state.arm.position`, `observation.state.effector.position`, `observation.images.*`). The `data_name` CLI flag must match the config filename stem.

3. **Norm stats JSON** (`assets/norm_stats/<name>.json`) — precomputed per-feature statistics. `norm_type` controls the method: `meanstd` for real-world, `bounds_99` for simulation.

4. **`VLADataset`** (`lingbotvla/data/vla_data/`) — assembles tokenized observations + images + action chunks using the robot config and norm stats. `chunk_size` is inherited from `train.chunk_size`.

### Configuration System

All hyperparameters live in YAML files parsed into three dataclasses in `lingbotvla/utils/arguments.py`: `ModelArguments`, `DataArguments`, `TrainingArguments`. Any field can be overridden on the CLI with `--group.field value`. The YAML is loaded first; CLI flags win.

Key training configs in `configs/vla/`:
- `real_load20000h.yaml` — real-robot: `norm_type: meanstd`, `loss_type: fm` (MSE), `lr: 5e-5`
- `robotwin_load20000h.yaml` — simulation: `norm_type: bounds_99`, `loss_type: L1_fm`, `lr: 1e-4`

### Training Loop (`tasks/vla/train_lingbotvla.py`)

Orchestrates: model build → FSDP/DDP wrap → dataset/dataloader → optimizer + cosine/constant LR scheduler → gradient-accumulation training loop → checkpoint saves. Checkpoints are saved as distributed checkpoint (DCP) format under `output/checkpoints/global_step_N/`; HF-format weights are written to `output/checkpoints/global_step_N/hf_ckpt/` when `save_hf_weights: true`.

**Inference requires**: `*.safetensors` + `config.json` + `lingbotvla_cli.yaml` in the checkpoint directory.

### Distributed Training

`lingbotvla/distributed/` supports DDP, FSDP1, and FSDP2. For fine-tuning, `data_parallel_mode: fsdp2` with `enable_full_shard: false` is the standard choice.

### Depth Variant

When using the depth-distilled model (`lingbot-vla-4b-depth`), set `align_params` in the YAML and provide `--model.moge_path` and `--model.morgbd_path`. The depth branch adds a cross-attention module between depth features and VLM tokens; training adds a contrastive + depth regression loss on top of the flow-matching loss.

## Code Style

- Line length 119, double quotes (enforced by `ruff`)
- Logging: use `helper.create_logger(__name__)` and `logger.info_rank0(...)` — this suppresses duplicate output from non-rank-0 processes

## Adding a Custom Robot

1. Create `configs/robot_configs/<robot_name>.yaml` following the `robotwin.yaml` pattern — map raw LeRobot keys to `arm.position`, `effector.position`, and `observation.images.*`.
2. Run `scripts/compute_norm.py` with `--data.data_name <robot_name>` to produce `assets/norm_stats/<robot_name>.json`.
3. Copy the closest base YAML from `configs/vla/`, set `data_name`, `norm_stats_file`, and adjust `joints`/`cameras` lists.
4. Launch with `bash train.sh`.
