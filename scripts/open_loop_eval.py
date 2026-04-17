"""
Open-loop evaluation: feed dataset observations into trained model,
compare predicted actions vs ground truth.
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
from lingbotvla.models import build_processor
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
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


def run_open_loop(model, episodes, dataset, norm_stats_file, norm_type, output_dir):
    """Run model on each frame and compare with GT actions."""
    os.makedirs(output_dir, exist_ok=True)

    # Build normalizer matching training config
    with open(norm_stats_file) as f:
        norm_stats = json.load(f)
    normalizer = Normalizer(
        norm_stats=norm_stats['norm_stats'],
        from_file=True,
        data_type='customized',
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
            # Ground truth action (raw, unnormalized)
            gt_action = item['action'].numpy()
            gt_actions.append(gt_action)

            # Build observation dict matching deployment format
            # Images: LeRobot gives C,H,W float [0,1] tensors
            cam_high = item['observation.images.cam_high'].numpy()       # C,H,W
            cam_left = item['observation.images.cam_left_wrist'].numpy()
            cam_right = item['observation.images.cam_right_wrist'].numpy()

            state = item['observation.state'].numpy()
            task = dataset.meta.tasks[int(item['task_index'])]

            observation = {
                'observation.images.cam_high': cam_high,        # C,H,W [0,1]
                'observation.images.cam_left_wrist': cam_left,
                'observation.images.cam_right_wrist': cam_right,
                'observation.state': state,
                'task': task,
            }

            # Normalize
            normalized_obs = normalizer.normalize(observation)

            # Convert to tensors if numpy
            for k, v in normalized_obs.items():
                if isinstance(v, np.ndarray):
                    normalized_obs[k] = torch.from_numpy(v)

            # Prepare model inputs (same as CustomizedRobotwinDataset.getdata)
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

            # Run model — call sample_actions directly to avoid
            # select_action's state/action reordering which is for robotwin_rep only
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
                # Take first 14 dims (actual action_dim) without reordering
                action_dim = item['action'].shape[-1]
                pred_action_chunk = actions.squeeze(0)[:, :action_dim].float().cpu().numpy()

            # Take first action from chunk as the prediction for this timestep
            pred_action_t = pred_action_chunk[0]

            # Unnormalize predicted action
            pred_action_norm = {'action': torch.tensor(pred_action_t)}
            pred_action_unnorm = normalizer.unnormalize(pred_action_norm)
            pred_actions.append(pred_action_unnorm['action'].numpy())

            if t % 10 == 0:
                print(f"  frame {t}/{len(frames)}")

        gt_actions = np.array(gt_actions)      # (T, action_dim)
        pred_actions = np.array(pred_actions)   # (T, action_dim)

        # Compute errors
        l1_error = np.abs(gt_actions - pred_actions).mean(axis=0)
        l2_error = np.sqrt(((gt_actions - pred_actions) ** 2).mean(axis=0))
        mean_l1 = l1_error.mean()
        mean_l2 = l2_error.mean()
        all_errors[ep_idx] = {'l1': mean_l1, 'l2': mean_l2}
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
        fig.suptitle(f'Episode {ep_idx} — Open-Loop Eval (L1={mean_l1:.4f})', fontsize=13)
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
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to hf_ckpt directory')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to LeRobot dataset')
    parser.add_argument('--norm_stats_file', type=str, required=True)
    parser.add_argument('--norm_type', type=str, default='bounds_99_woclip')
    parser.add_argument('--episodes', type=str, default='0,1,2,3,4', help='Comma-separated episode indices')
    parser.add_argument('--output_dir', type=str, default='open_loop_results')
    args = parser.parse_args()

    episode_indices = [int(x) for x in args.episodes.split(',')]

    # Load model
    print("Loading model...")
    model = QwenPiServer(
        path_to_pi_model=args.ckpt_path,
        use_length=1,
        chunk_ret=False,
        use_bf16=True,
    )

    # Override normalizer with correct one matching training
    with open(args.norm_stats_file) as f:
        norm_stats = json.load(f)
    model.vla.normalizer = Normalizer(
        norm_stats=norm_stats['norm_stats'],
        from_file=True,
        data_type='customized',
        norm_type={
            "observation.images.cam_high": "identity",
            "observation.images.cam_left_wrist": "identity",
            "observation.images.cam_right_wrist": "identity",
            "observation.state": args.norm_type,
            "action": args.norm_type,
        },
    )

    # Load dataset episodes
    print("Loading episodes...")
    episodes, dataset = load_episodes(args.dataset_path, episode_indices)

    # Run evaluation
    run_open_loop(model, episodes, dataset, args.norm_stats_file, args.norm_type, args.output_dir)


if __name__ == '__main__':
    main()
