Walk the user through fine-tuning LingBot-VLA on their custom dataset, end to end. Follow these steps exactly in order, asking the user for required information at each step before proceeding.

---

## Step 0 — Gather inputs

Ask the user for the following (can be provided all at once or one at a time):

1. **Dataset path** — path to the LeRobot-format dataset directory
2. **Dataset name** — a short identifier (snake_case, e.g. `my_task`) used as `data_name` in the YAML config and the Python branch
3. **Action dimension** — the actual DOF of the robot (e.g. 14 for a dual-arm with grippers)
4. **State dimension** — typically same as action dim
5. **LeRobot action key** — the key in the dataset for actions (e.g. `action`)
6. **LeRobot state key** — the key for proprioceptive state (e.g. `observation.state`)
7. **Image keys** — list of camera keys in the dataset (e.g. `observation.images.cam_high`, `observation.images.cam_left_wrist`, `observation.images.cam_right_wrist`)
8. **Model checkpoint path** — path to the pretrained LingBot-VLA checkpoint (e.g. `ckpts/lingbot-vla-4b`)
9. **Tokenizer path** — path to `Qwen2.5-VL-3B-Instruct`
10. **Output directory** — where checkpoints and logs will be saved
11. **Number of GPUs** — used to set `global_batch_size` = `micro_batch_size × num_gpus`
12. **Micro batch size** — per-GPU batch size (default: 8)

---

## Step 1 — Verify data format

Read the dataset's `meta/info.json` (or `meta/stats.json` if present) and confirm that the action key, state key, and image keys the user provided actually exist in the dataset. Report what you find. If keys are missing, tell the user which keys are present so they can correct their input.

---

## Step 2 — Compute normalization stats

Check whether `assets/norm_stats/{dataset_name}.json` already exists.

If it does not, tell the user to run:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_robotwin_5.py configs/norm/robotwin_5.yaml \
  --model.model_path {model_path} \
  --model.tokenizer_path {tokenizer_path} \
  --data.train_path {dataset_path} \
  --data.norm_path assets/norm_stats/{dataset_name}.json
```

Also check lines 73–75 of `scripts/compute_norm_robotwin_5.py` to confirm `state_norm_keys` and `acton_norm_keys` match the user's keys. If they differ, show the user exactly what to change:
- `state_norm_keys = ['{state_key}']`
- `acton_norm_keys = ['{action_key}']`
- `delta_norm = {'{action_key}': False}`

Wait for the user to confirm the norm stats file has been generated before continuing.

---

## Step 3 — Create dataset class

Read `lingbotvla/data/vla_data/base_dataset.py` and examine `CustomizedRobotwinDataset` as the template.

Create a new dataset class named `{DatasetName}Dataset` (CamelCase of `dataset_name`) by:
- Copying the structure of `CustomizedRobotwinDataset`
- Setting `data_type='customized'` in the `Normalizer` constructor
- Replacing image key names with the user's actual image keys
- Replacing `observation.state` and `action` with the user's actual state/action keys
- Setting `delta_timestamps` using the user's action key

Add the new class to the bottom of `lingbotvla/data/vla_data/base_dataset.py`.

Then register it in `lingbotvla/data/vla_data/__init__.py`:
```python
from .base_dataset import ..., {DatasetName}Dataset
```

---

## Step 4 — Add training branch

Read `tasks/vla/train_lingbotvla.py` and find the dataset selection block (around line 328–335). Add a new `elif` branch:

```python
elif args.data.data_name == '{dataset_name}':
    train_dataset = {DatasetName}Dataset(
        repo_id=args.data.train_path,
        config=model.config,
        tokenizer=processor.tokenizer,
        data_config=args.data,
        image_processor=processor.image_processor if 'qwen' in args.model.tokenizer_path.lower() else None,
        use_depth_align=use_depth_align,
    )
```

---

## Step 5 — Create YAML config

Create `configs/vla/{dataset_name}_finetune.yaml` with the following content, filled in with the user's values:

```yaml
model:
  model_path: {model_path}
  tokenizer_path: {tokenizer_path}
  post_training: true
  adanorm_time: true
  old_adanorm: true

data:
  datasets_type: vla
  data_name: {dataset_name}
  train_path: {dataset_path}
  num_workers: 8
  norm_type: bounds_99_woclip
  norm_stats_file: assets/norm_stats/{dataset_name}.json

train:
  output_dir: {output_dir}
  loss_type: L1_fm
  data_parallel_mode: fsdp2
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true
  use_wandb: false
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false
  tokenizer_max_length: 24
  action_dim: {action_dim}
  max_action_dim: 75
  max_state_dim: 75
  lr: 1.0e-4
  lr_decay_style: constant
  num_train_epochs: 69
  micro_batch_size: {micro_batch_size}
  global_batch_size: {global_batch_size}
  ckpt_manager: dcp
  save_steps: 10000
  save_hf_weights: true
  save_epochs: 69
  enable_fp32: true
  enable_resume: true
```

---

## Step 6 — Show the final training command

Print the ready-to-run command:

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/{dataset_name}_finetune.yaml
```

Remind the user:
- Logs are tee'd to `log.txt`
- TensorBoard: `tensorboard --logdir {output_dir}/runs`
- Checkpoints: `{output_dir}/checkpoints/global_step_N/`
- HF weights (for deploy): `{output_dir}/checkpoints/global_step_N/hf_ckpt/`
- Training auto-resumes from the latest checkpoint if restarted

---

## Step 7 — Deployment reminder

Remind the user that the deploy server must use the **same image key names and `data_type='customized'`** as the dataset class. Point them to `deploy/lingbot_robotwin_policy.py` line 323 as the reference for where to set `data_type` in the normalizer construction.

The deploy command template:
```bash
export QWEN25_PATH={tokenizer_path}
python -m deploy.lingbot_robotwin_policy \
  --model_path {output_dir}/checkpoints/global_step_N/hf_ckpt \
  --use_length 50 \
  --port 8765
```
