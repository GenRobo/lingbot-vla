"""
Open-loop evaluation for the provided pretrained checkpoint (lingbot-vla-4b-posttrain-robotwin).

This checkpoint was trained with:
  - RobotwinDataset (data_type='robotwin' in Normalizer)
  - norm_stats: robotwin_all_new.json (split keys: action.arm.position, etc.)
  - norm_type: bounds_99_woclip
  - Joint reordering: robot order -> model order in infer(), reverse in select_action()

The deployment code (lingbot_robotwin_policy_rep.py) uses data_type='robotwin_rep'
which differs slightly from training's 'robotwin' in state assembly (no reorder for state).
We use 'robotwin_rep' here to match the deployment normalizer exactly.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../deploy'))

from lingbotvla.data.vla_data.transform import Normalizer, prepare_images, prepare_language, prepare_state
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from deploy.lingbot_robotwin_policy_rep import QwenPiServer


def load_episodes(dataset_path, episode_indices):
    """Load full episodes from LeRobot dataset."""
    dataset = LeRobotDataset(repo_id=dataset_path)
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
    return episodes, dataset


def run_open_loop(model, episodes, dataset, output_dir):
    """
    Run provided checkpoint on each frame and compare with GT actions.

    The provided checkpoint uses the default deployment normalizer:
      - data_type='robotwin_rep'
      - norm_stats from robotwin_all_new.json
      - norm_type: bounds_99_woclip

    The infer() method handles:
      1. Image resize to 224x224
      2. State reordering: robot [L6,Lgrip,R6,Rgrip] -> model [L6,R6,Lgrip,Rgrip]
      3. Normalization
      4. Model inference
      5. Action reordering back: model -> robot order
      6. Unnormalization

    We replicate this pipeline using QwenPiServer.infer() directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_errors = {}

    for ep_idx, frames in episodes.items():
        print(f"\n=== Episode {ep_idx} ({len(frames)} frames) ===")
        gt_actions_raw = []
        pred_actions_raw = []

        # Reset model
        model.global_step = 0
        model.last_action_chunk = None

        for t, item in enumerate(frames):
            # Ground truth action (raw, unnormalized, 14-dim)
            gt_action = item['action'].numpy()
            gt_actions_raw.append(gt_action)

            # Build observation dict as QwenPiServer.infer() expects:
            # Images: H,W,C uint8 numpy arrays (infer does resize_image internally)
            # State: 14-dim numpy array in robot order [L6, Lgrip, R6, Rgrip]
            cam_high = item['observation.images.cam_high']       # C,H,W tensor [0,1]
            cam_left = item['observation.images.cam_left_wrist']
            cam_right = item['observation.images.cam_right_wrist']

            # Convert C,H,W float [0,1] -> H,W,C uint8 (what the robot would provide)
            cam_high_np = (cam_high.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cam_left_np = (cam_left.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cam_right_np = (cam_right.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            state = item['observation.state'].numpy()  # 14-dim
            task_str = dataset.meta.tasks[int(item['task_index'])]

            observation = {
                'observation.images.cam_high': cam_high_np,
                'observation.images.cam_left_wrist': cam_left_np,
                'observation.images.cam_right_wrist': cam_right_np,
                'observation.state': state,
                'task': task_str,
            }

            # Use QwenPiServer.infer() which handles all reordering + normalization
            result = model.infer(observation)
            pred_action = result['action']  # numpy, shape (use_length, action_dim) or (action_dim,)

            if pred_action.ndim == 2:
                pred_action = pred_action[0]  # take first step

            pred_actions_raw.append(pred_action[:gt_action.shape[-1]])

            if t % 10 == 0:
                print(f"  frame {t}/{len(frames)}")

        gt_actions = np.array(gt_actions_raw)      # (T, 14)
        pred_actions = np.array(pred_actions_raw)   # (T, 14)

        # Compute errors
        l1_error = np.abs(gt_actions - pred_actions).mean(axis=0)
        l2_error = np.sqrt(((gt_actions - pred_actions) ** 2).mean(axis=0))
        mean_l1 = l1_error.mean()
        mean_l2 = l2_error.mean()
        all_errors[ep_idx] = {'l1': float(mean_l1), 'l2': float(mean_l2)}
        print(f"  Mean L1: {mean_l1:.4f}, Mean L2: {mean_l2:.4f}")
        print(f"  Per-joint L1: {np.round(l1_error, 4).tolist()}")

        # Plot
        action_dim = gt_actions.shape[1]
        joint_names = [
            'L_waist', 'L_shoulder', 'L_elbow', 'L_forearm', 'L_wrist_a', 'L_wrist_r', 'L_gripper',
            'R_waist', 'R_shoulder', 'R_elbow', 'R_forearm', 'R_wrist_a', 'R_wrist_r', 'R_gripper',
        ][:action_dim]

        fig, axes = plt.subplots(action_dim, 1, figsize=(12, 2.5 * action_dim), sharex=True)
        if action_dim == 1:
            axes = [axes]
        for j in range(action_dim):
            axes[j].plot(gt_actions[:, j], 'b-', label='GT', linewidth=1.5)
            axes[j].plot(pred_actions[:, j], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
            axes[j].set_ylabel(joint_names[j], fontsize=9)
            axes[j].legend(loc='upper right', fontsize=8)
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Timestep')
        fig.suptitle(f'Episode {ep_idx} — Open-Loop Eval Provided Ckpt (L1={mean_l1:.4f})', fontsize=13)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'episode_{ep_idx}.png')
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  Plot saved: {plot_path}")

    # Summary
    print("\n=== Summary ===")
    for ep_idx, err in all_errors.items():
        print(f"  Episode {ep_idx}: L1={err['l1']:.4f}, L2={err['l2']:.4f}")
    avg_l1 = np.mean([e['l1'] for e in all_errors.values()])
    avg_l2 = np.mean([e['l2'] for e in all_errors.values()])
    print(f"  Average: L1={avg_l1:.4f}, L2={avg_l2:.4f}")

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({str(k): v for k, v in all_errors.items()}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='ckpts/lingbot-vla-4b-posttrain-robotwin',
                        help='Path to provided checkpoint directory')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to LeRobot dataset (e.g. click_bell_all550)')
    parser.add_argument('--episodes', type=str, default='0,1,2,3,4',
                        help='Comma-separated episode indices')
    parser.add_argument('--output_dir', type=str,
                        default='open_loop_results_provided')
    args = parser.parse_args()

    episode_indices = [int(x) for x in args.episodes.split(',')]

    # Load model using QwenPiServer — it will use the default normalizer
    # (data_type='robotwin_rep', norm_stats='robotwin_all_new.json')
    # which matches this checkpoint's training setup
    print("Loading model...")
    model = QwenPiServer(
        path_to_pi_model=args.ckpt_path,
        use_length=1,
        chunk_ret=True,
        use_bf16=True,
    )

    # Load dataset episodes
    print("Loading episodes...")
    episodes, dataset = load_episodes(args.dataset_path, episode_indices)

    # Run evaluation
    run_open_loop(model, episodes, dataset, args.output_dir)


if __name__ == '__main__':
    main()
