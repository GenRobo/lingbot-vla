"""
Open-loop evaluation for BM-100 fine-tuned checkpoint.

Differences from robotwin open-loop eval:
  - Split keys (left_arm, right_arm, left_gripper, right_gripper) concatenated to 16D
  - data_type='bm100' in Normalizer
  - No joint reordering (unlike robotwin's select_action)
  - Static dim clamping on state input (matching training pipeline)
  - Image key mapping: head_rgb -> cam_high, etc.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# QWEN25_PATH must be set before importing QwenPiServer (it reads env at import time).
# Export it in your shell: export QWEN25_PATH=/path/to/Qwen2.5-VL-3B-Instruct
if not os.environ.get('QWEN25_PATH'):
    raise EnvironmentError("QWEN25_PATH is not set. Export it before running this script.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../deploy'))

from lingbotvla.data.vla_data.transform import Normalizer, prepare_images, prepare_language, prepare_state
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from deploy.lingbot_robotwin_policy_rep import QwenPiServer

# Must match BM100Dataset in base_dataset.py
BM100_STATE_KEYS = [
    "observation.state.left_arm",      # (7,)
    "observation.state.right_arm",     # (7,)
    "observation.state.left_gripper",  # scalar
    "observation.state.right_gripper", # scalar
]
BM100_ACTION_KEYS = [
    "action.left_arm",      # (7,)
    "action.right_arm",     # (7,)
    "action.left_gripper",  # scalar
    "action.right_gripper", # scalar
]


def concat_split_keys(item, keys):
    """Concatenate split keys into a single tensor."""
    parts = []
    for k in keys:
        v = item[k]
        if v.dim() == 0:
            v = v.unsqueeze(0)
        parts.append(v)
    return torch.cat(parts, dim=-1)


def build_static_dim_info(norm_stats_file):
    """Build static dim masks and means from norm stats (same logic as BM100Dataset)."""
    with open(norm_stats_file) as f:
        norm_stats = json.load(f)['norm_stats']

    static_info = {}
    static_threshold = 0.01
    for key in ("observation.state", "action"):
        q01 = np.array(norm_stats[key]['q01'])
        q99 = np.array(norm_stats[key]['q99'])
        mean = np.array(norm_stats[key]['mean'])
        mask = (q99 - q01) < static_threshold
        static_info[key] = {'mask': mask, 'mean': mean}
        if mask.any():
            dim_names = []
            offset = 0
            for name, size in [("left_arm", 7), ("right_arm", 7), ("left_gripper", 1), ("right_gripper", 1)]:
                for d in range(size):
                    if mask[offset + d]:
                        dim_names.append(f"{name}[{d}]")
                offset += size
            print(f"[Static dims] {key}: {dim_names}")
    return static_info


def clamp_static(tensor, key, static_info):
    """Replace static dims with their mean value."""
    mask = static_info[key]['mask']
    mean = static_info[key]['mean']
    mean_t = torch.from_numpy(mean).to(dtype=tensor.dtype)
    tensor[mask] = mean_t[mask]
    return tensor


def load_episodes(dataset_path, episode_indices):
    """Load full episodes from BM-100 LeRobot dataset."""
    root_path = Path(dataset_path)
    if root_path.is_dir():
        ds_kwargs = {"repo_id": root_path.name, "root": str(root_path)}
    else:
        ds_kwargs = {"repo_id": dataset_path}

    dataset = LeRobotDataset(**ds_kwargs)
    meta = dataset.meta

    episodes = {}
    for ep_idx in episode_indices:
        ep_frames = []
        for i in range(len(dataset)):
            item = dataset[i]
            if int(item['episode_index']) == ep_idx:
                ep_frames.append(item)
            elif int(item['episode_index']) > ep_idx and len(ep_frames) > 0:
                break
        if ep_frames:
            episodes[ep_idx] = ep_frames
            print(f"Episode {ep_idx}: {len(ep_frames)} frames")
    return episodes, dataset, meta


def run_open_loop(model, episodes, meta, norm_stats_file, norm_type, output_dir, action_dim):
    """Run model on each frame and compare with GT actions."""
    os.makedirs(output_dir, exist_ok=True)

    static_info = build_static_dim_info(norm_stats_file)

    # Build normalizer matching training config
    with open(norm_stats_file) as f:
        norm_stats = json.load(f)
    normalizer = Normalizer(
        norm_stats=norm_stats['norm_stats'],
        from_file=True,
        data_type='bm100',
        norm_type={
            "observation.images.cam_high": "identity",
            "observation.images.cam_left_wrist": "identity",
            "observation.images.cam_right_wrist": "identity",
            "observation.state": norm_type,
            "action": norm_type,
        },
    )

    all_errors = {}

    for ep_idx, frames in episodes.items():
        print(f"\n=== Episode {ep_idx} ({len(frames)} frames) ===")
        gt_actions = []
        pred_actions = []

        # Reset model state
        model.global_step = 0
        model.last_action_chunk = None

        for t, item in enumerate(frames):
            # Ground truth action: concatenate split keys (raw, unnormalized)
            gt_action = concat_split_keys(item, BM100_ACTION_KEYS).numpy()
            gt_actions.append(gt_action)

            # Build state: concatenate split keys + clamp static dims
            state = concat_split_keys(item, BM100_STATE_KEYS)
            state = clamp_static(state, "observation.state", static_info)

            # Map image keys: BM-100 uses head_rgb, left_wrist_rgb, right_wrist_rgb
            # Convert C,H,W float [0,1] -> H,W,C uint8 -> resize -> C,H,W float [0,1]
            def process_image(img_tensor, target_size=224):
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil = Image.fromarray(img_np).resize((target_size, target_size), Image.BILINEAR)
                return np.transpose(np.array(pil), (2, 0, 1)) / 255.0

            cam_high = process_image(item['observation.images.head_rgb'])
            cam_left = process_image(item['observation.images.left_wrist_rgb'])
            cam_right = process_image(item['observation.images.right_wrist_rgb'])

            task = meta.tasks[int(item['task_index'])]

            observation = {
                'observation.images.cam_high': cam_high,
                'observation.images.cam_left_wrist': cam_left,
                'observation.images.cam_right_wrist': cam_right,
                'observation.state': state.numpy(),
                'task': task,
            }

            # Normalize
            normalized_obs = normalizer.normalize(observation)

            # Convert to tensors
            for k, v in normalized_obs.items():
                if isinstance(v, np.ndarray):
                    normalized_obs[k] = torch.from_numpy(v)

            # Prepare model inputs
            base_image = (normalized_obs["observation.images.cam_high"] * 255).to(torch.uint8)
            left_image = (normalized_obs["observation.images.cam_left_wrist"] * 255).to(torch.uint8)
            right_image = (normalized_obs["observation.images.cam_right_wrist"] * 255).to(torch.uint8)

            obs_dict = {
                "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_image, "right_wrist_0_rgb": right_image},
                "state": normalized_obs["observation.state"].to(torch.float32),
                "prompt": [task],
            }

            state_tensor = prepare_state(model.config, obs_dict)
            lang_tokens, lang_masks = prepare_language(model.config, model.language_tokenizer, obs_dict)
            images, img_masks, _ = prepare_images(model.config, model.image_processor, obs_dict)

            model_input = {
                'images': images,
                'img_masks': img_masks,
                'state': state_tensor,
                'lang_tokens': lang_tokens,
                'lang_masks': lang_masks,
            }

            dtype = torch.bfloat16 if model.use_bf16 else torch.float32
            if model.use_bf16:
                model_input['state'] = model_input['state'].to(torch.bfloat16)

            # Run model — call sample_actions directly, NO joint reordering for BM-100
            with torch.no_grad():
                if len(model_input['images'].shape) == 4:
                    model_input['images'] = model_input['images'].unsqueeze(0)
                    model_input['img_masks'] = model_input['img_masks'].unsqueeze(0)
                actions = model.vla.model.sample_actions(
                    model_input['images'].to(dtype=dtype, device='cuda'),
                    model_input['img_masks'].to(device='cuda'),
                    model_input['lang_tokens'].unsqueeze(0).to(device='cuda'),
                    model_input['lang_masks'].unsqueeze(0).to(device='cuda'),
                    model_input['state'].unsqueeze(0).to(dtype=dtype, device='cuda'),
                    vlm_causal=model.config.vlm_causal,
                )
                # actions shape: (1, chunk_size, max_action_dim)
                # Take first action_dim dims WITHOUT reordering
                pred_action_chunk = actions.squeeze(0)[:, :action_dim].float().cpu()

            # Take first action from chunk
            pred_action_t = pred_action_chunk[0]

            # Unnormalize predicted action
            pred_unnorm = normalizer.unnormalize({'action': pred_action_t})
            pred_actions.append(pred_unnorm['action'].numpy())

            if t % 10 == 0:
                print(f"  frame {t}/{len(frames)}")

        gt_actions = np.array(gt_actions)      # (T, 16)
        pred_actions = np.array(pred_actions)   # (T, 16)

        # Compute errors
        l1_error = np.abs(gt_actions - pred_actions).mean(axis=0)
        l2_error = np.sqrt(((gt_actions - pred_actions) ** 2).mean(axis=0))
        mean_l1 = l1_error.mean()
        mean_l2 = l2_error.mean()

        # Compute errors for active dims only: right_arm(7-13) + right_gripper(15)
        active_dims = list(range(7, 14)) + [15]
        right_l1 = l1_error[active_dims].mean()
        right_l2 = l2_error[active_dims].mean()

        all_errors[ep_idx] = {
            'l1': float(mean_l1), 'l2': float(mean_l2),
            'right_l1': float(right_l1), 'right_l2': float(right_l2),
        }
        print(f"  All dims  — L1: {mean_l1:.4f}, L2: {mean_l2:.4f}")
        print(f"  Right arm — L1: {right_l1:.4f}, L2: {right_l2:.4f}")
        print(f"  Per-joint L1: {np.round(l1_error, 4).tolist()}")

        # Plot
        # Dim order: left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
        joint_names = [
            'L_j0', 'L_j1', 'L_j2', 'L_j3', 'L_j4', 'L_j5', 'L_j6',
            'R_j0', 'R_j1', 'R_j2', 'R_j3', 'R_j4', 'R_j5', 'R_j6',
            'L_grip', 'R_grip',
        ][:action_dim]

        fig, axes = plt.subplots(action_dim, 1, figsize=(14, 2.2 * action_dim), sharex=True)
        if action_dim == 1:
            axes = [axes]
        for j in range(action_dim):
            axes[j].plot(gt_actions[:, j], 'b-', label='GT', linewidth=1.5)
            axes[j].plot(pred_actions[:, j], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
            axes[j].set_ylabel(joint_names[j], fontsize=9)
            axes[j].legend(loc='upper right', fontsize=8)
            axes[j].grid(True, alpha=0.3)
            # Highlight static dims (left_arm 0-6, left_gripper 14)
            if j < 7 or j == 14:
                axes[j].set_facecolor('#f5f5f5')
        axes[-1].set_xlabel('Timestep')
        fig.suptitle(f'Episode {ep_idx} — BM-100 Open-Loop (All L1={mean_l1:.4f}, Right L1={right_l1:.4f})', fontsize=13)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'episode_{ep_idx}.png')
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  Plot saved: {plot_path}")

    # Summary
    print("\n=== Summary ===")
    for ep_idx, err in all_errors.items():
        print(f"  Episode {ep_idx}: All L1={err['l1']:.4f}, L2={err['l2']:.4f} | Right L1={err['right_l1']:.4f}, L2={err['right_l2']:.4f}")
    avg_l1 = np.mean([e['l1'] for e in all_errors.values()])
    avg_l2 = np.mean([e['l2'] for e in all_errors.values()])
    avg_right_l1 = np.mean([e['right_l1'] for e in all_errors.values()])
    avg_right_l2 = np.mean([e['right_l2'] for e in all_errors.values()])
    print(f"  Average — All: L1={avg_l1:.4f}, L2={avg_l2:.4f} | Right: L1={avg_right_l1:.4f}, L2={avg_right_l2:.4f}")

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({str(k): v for k, v in all_errors.items()}, f, indent=2)


def prepare_ckpt(ckpt_path, cli_yaml_path):
    """Ensure lingbotvla_cli.yaml is in the hf_ckpt directory for QwenPiServer."""
    ckpt_yaml = os.path.join(ckpt_path, 'lingbotvla_cli.yaml')
    if not os.path.exists(ckpt_yaml):
        import shutil
        shutil.copy2(cli_yaml_path, ckpt_yaml)
        print(f"Copied {cli_yaml_path} -> {ckpt_yaml}")


def main():
    parser = argparse.ArgumentParser(description='BM-100 Open-Loop Evaluation')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to hf_ckpt directory (with safetensors)')
    parser.add_argument('--cli_yaml', type=str, default=None,
                        help='Path to lingbotvla_cli.yaml (auto-detected from training output_dir if not set)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to BM-100 LeRobot dataset')
    parser.add_argument('--norm_stats_file', type=str,
                        default='assets/norm_stats/bm100.json')
    parser.add_argument('--norm_type', type=str, default='bounds_99_woclip')
    parser.add_argument('--episodes', type=str, default='0,1,2,3,4',
                        help='Comma-separated episode indices')
    parser.add_argument('--output_dir', type=str, default='open_loop_results_bm100')
    args = parser.parse_args()

    episode_indices = [int(x) for x in args.episodes.split(',')]

    # Resolve cli_yaml path
    ckpt_path = Path(args.ckpt_path)
    if args.cli_yaml:
        cli_yaml = args.cli_yaml
    else:
        # Try to find it relative to the checkpoint
        # e.g., .../checkpoints/global_step_50000/hf_ckpt -> .../lingbotvla_cli.yaml
        for parent in ckpt_path.parents:
            candidate = parent / 'lingbotvla_cli.yaml'
            if candidate.exists():
                cli_yaml = str(candidate)
                break
        else:
            raise FileNotFoundError(
                f"Could not find lingbotvla_cli.yaml near {ckpt_path}. Pass --cli_yaml explicitly.")

    # Ensure cli.yaml is in the ckpt dir
    prepare_ckpt(str(ckpt_path), cli_yaml)

    # Load model
    print("Loading model...")
    model = QwenPiServer(
        path_to_pi_model=str(ckpt_path),
        use_length=1,
        chunk_ret=False,
        use_bf16=True,
    )

    # Override normalizer with BM-100 config (QwenPiServer defaults to robotwin)
    with open(args.norm_stats_file) as f:
        norm_stats = json.load(f)
    model.vla.normalizer = Normalizer(
        norm_stats=norm_stats['norm_stats'],
        from_file=True,
        data_type='bm100',
        norm_type={
            "observation.images.cam_high": "identity",
            "observation.images.cam_left_wrist": "identity",
            "observation.images.cam_right_wrist": "identity",
            "observation.state": args.norm_type,
            "action": args.norm_type,
        },
    )
    action_dim = model.action_dim  # Should be 16 from training config

    # Load dataset episodes
    print("Loading episodes...")
    episodes, dataset, meta = load_episodes(args.dataset_path, episode_indices)

    # Run evaluation
    run_open_loop(model, episodes, meta, args.norm_stats_file, args.norm_type,
                  args.output_dir, action_dim)


if __name__ == '__main__':
    main()
