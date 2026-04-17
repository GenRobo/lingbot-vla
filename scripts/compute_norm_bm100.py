"""Compute normalization statistics for BM-100 dataset.

Concatenates split state/action sub-keys into 16D vectors:
  State: left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
  Action: action.left_arm(7) + action.right_arm(7) + action.left_gripper(1) + action.right_gripper(1)
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import asdict, dataclass, field

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lingbotvla.utils import helper
from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
import lingbotvla.utils.normalize as normalize

logger = helper.create_logger(__name__)

STATE_KEYS = [
    "observation.state.left_arm",      # 7D
    "observation.state.right_arm",      # 7D
    "observation.state.left_gripper",   # scalar
    "observation.state.right_gripper",  # scalar
]
ACTION_KEYS = [
    "action.left_arm",      # 7D
    "action.right_arm",     # 7D
    "action.left_gripper",  # scalar
    "action.right_gripper", # scalar
]


@dataclass
class MyDataArguments(DataArguments):
    norm_path: str = field(
        default=None,
        metadata={"help": "Path to save norm stats."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def ensure_2d(t):
    """Ensure batched tensors are at least 2D: (B,) -> (B,1)."""
    if t.dim() == 1:
        return t.unsqueeze(-1)
    return t


def compute_norm(dataset, batch_size, stats):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8,
        shuffle=False, drop_last=False,
    )
    success = True
    try:
        for batch in tqdm(data_loader, desc="Computing BM-100 norm stats"):
            # State: (B, 16)
            state_parts = [ensure_2d(batch[k]) for k in STATE_KEYS]
            state = torch.cat(state_parts, dim=-1).numpy()
            stats["observation.state"].update(state.reshape(-1, state.shape[-1]))

            # Action: (B, 16)
            action_parts = [ensure_2d(batch[k]) for k in ACTION_KEYS]
            action = torch.cat(action_parts, dim=-1).numpy()
            stats["action"].update(action.reshape(-1, action.shape[-1]))
    except Exception as e:
        import traceback
        traceback.print_exc()
        success = False
    return success


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    logger.info_rank0("Prepare data")
    assert args.data.datasets_type == 'vla'

    # Load raw single-frame data (no delta_timestamps) for norm computation
    train_path = Path(args.data.train_path)
    if train_path.is_dir():
        dataset = LeRobotDataset(repo_id=train_path.name, root=str(train_path))
    else:
        dataset = LeRobotDataset(repo_id=args.data.train_path)

    norm_keys = ["observation.state", "action"]
    stats = {key: normalize.RunningStats() for key in norm_keys}

    success = compute_norm(dataset, args.train.global_batch_size, stats)

    if success:
        norm_stats = {key: s.get_statistics() for key, s in stats.items()}
        output_path = Path(args.data.norm_path)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats, stats["observation.state"]._count)
    else:
        print("Norm computation failed!")


if __name__ == "__main__":
    main()
