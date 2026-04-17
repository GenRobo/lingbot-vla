# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Callable, Dict, List, Literal, Optional
import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer, AutoImageProcessor
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import json
import yaml
from PIL import Image
from .transform import Normalizer, prepare_action, prepare_images, prepare_language, prepare_state

from ...utils import logging

class VlaDataset(Dataset):
    def __init__(
        self,
        repo_id="path2dataset",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        action_name="action",
    ):
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            action_name: [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
        )
        self.action_name = action_name

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']
        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )

class liberoDataset(Dataset):
    def __init__(
        self,
        repo_id="libero",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "actions": [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        self.normalizer = Normalizer(
            # norm_stats=self.dataset.meta.stats,
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='libero',
            norm_type={
                "image": "identity",
                "wrist_image": "identity",
                "state": data_config.norm_type,
                "actions": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["image"] * 255).to(torch.uint8)
        wrist_image = (normalized_item["wrist_image"] * 255).to(
            torch.uint8
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": wrist_image},
            "state": normalized_item["state"].to(torch.float32),
            "action": normalized_item["actions"].to(torch.float32),
            "action_is_pad": normalized_item["actions_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict,  use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }

        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

class RobotwinDataset(Dataset):
    def __init__(
        self,
        repo_id="robotwin",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        self.normalizer = Normalizer(
            # norm_stats=self.dataset.meta.stats,
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='robotwin',
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (normalized_item["observation.images.cam_left_wrist"] * 255).to(
            torch.uint8
        )
        right_wrist_image = (normalized_item["observation.images.cam_right_wrist"] * 255).to(
            torch.uint8
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )


# ── BM-100 (Galaxea R1 Pro) ─────────────────────────────────────────
# Split state/action sub-keys are concatenated into 16D vectors:
#   State:  left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
#   Action: left_arm(7) + right_arm(7) + left_gripper(1) + right_gripper(1)
# Images:  head_rgb → base, left_wrist_rgb → left_wrist, right_wrist_rgb → right_wrist

BM100_STATE_KEYS = [
    "observation.state.left_arm",      # (7,)
    "observation.state.right_arm",     # (7,)
    "observation.state.left_gripper",  # scalar
    "observation.state.right_gripper", # scalar
]
BM100_ACTION_KEYS = [
    "action.left_arm",      # (chunk, 7)
    "action.right_arm",     # (chunk, 7)
    "action.left_gripper",  # (chunk,)
    "action.right_gripper", # (chunk,)
]


class BM100Dataset(Dataset):
    def __init__(
        self,
        repo_id="path_to_bm100",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file

        from pathlib import Path
        train_path = Path(repo_id)
        if train_path.is_dir():
            ds_kwargs = {"repo_id": train_path.name, "root": str(train_path)}
        else:
            ds_kwargs = {"repo_id": repo_id}

        self.dataset_meta = LeRobotDatasetMetadata(**ds_kwargs)

        chunk_size = getattr(data_config, 'chunk_size', 50)
        delta_ts = [t / self.dataset_meta.fps for t in range(chunk_size)]
        delta_timestamps = {k: delta_ts for k in BM100_ACTION_KEYS}

        self.dataset = LeRobotDataset(
            **ds_kwargs,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        self.normalizer = Normalizer(
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='bm100',
            norm_type={
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

        # Detect static dims: where q99 - q01 < threshold in norm stats
        static_threshold = 0.01
        ns = self.norm_stats['norm_stats']
        for key in ("observation.state", "action"):
            q01 = np.array(ns[key]['q01'])
            q99 = np.array(ns[key]['q99'])
            mean = np.array(ns[key]['mean'])
            static_mask = (q99 - q01) < static_threshold
            setattr(self, f"_{key.replace('.', '_')}_static_mask", static_mask)
            setattr(self, f"_{key.replace('.', '_')}_static_mean", mean)
            if static_mask.any():
                dim_names = []
                offset = 0
                for k, size in [("left_arm", 7), ("right_arm", 7), ("left_gripper", 1), ("right_gripper", 1)]:
                    for d in range(size):
                        if static_mask[offset + d]:
                            dim_names.append(f"{k}[{d}]")
                    offset += size
                print(f"[BM100Dataset] Static dims in {key}: {dim_names}")

    def __len__(self):
        return len(self.dataset)

    def _clamp_static(self, tensor, key):
        """Replace static dims with their mean value."""
        mask = getattr(self, f"_{key.replace('.', '_')}_static_mask")
        mean = getattr(self, f"_{key.replace('.', '_')}_static_mean")
        mean_t = torch.from_numpy(mean).to(dtype=tensor.dtype)
        if tensor.dim() == 2:
            # action: (chunk, 16)
            tensor[:, mask] = mean_t[mask]
        else:
            # state: (16,)
            tensor[mask] = mean_t[mask]
        return tensor

    def _concat_state(self, item):
        """Concatenate split state keys into a single (16,) tensor."""
        parts = []
        for k in BM100_STATE_KEYS:
            v = item[k]
            if v.dim() == 0:
                v = v.unsqueeze(0)
            parts.append(v)
        return torch.cat(parts, dim=-1)

    def _concat_action(self, item):
        """Concatenate split action keys into a single (chunk, 16) tensor."""
        parts = []
        for k in BM100_ACTION_KEYS:
            v = item[k]
            if v.dim() == 1:
                v = v.unsqueeze(-1)  # (chunk,) -> (chunk, 1)
            parts.append(v)
        return torch.cat(parts, dim=-1)

    def _get_action_is_pad(self, item):
        """Use the first action sub-key's pad mask (they're all identical)."""
        return item[f"{BM100_ACTION_KEYS[0]}_is_pad"]

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

        # Concatenate split keys into single tensors and clamp static dims
        item["observation.state"] = self._clamp_static(self._concat_state(item), "observation.state")
        item["action"] = self._clamp_static(self._concat_action(item), "action")
        item["action_is_pad"] = self._get_action_is_pad(item)

        # Map image keys
        item["observation.images.cam_high"] = item["observation.images.head_rgb"]
        item["observation.images.cam_left_wrist"] = item["observation.images.left_wrist_rgb"]
        item["observation.images.cam_right_wrist"] = item["observation.images.right_wrist_rgb"]

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (normalized_item["observation.images.cam_left_wrist"] * 255).to(torch.uint8)
        right_wrist_image = (normalized_item["observation.images.cam_right_wrist"] * 255).to(torch.uint8)

        batch_dict = {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict)
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict)
        actions = prepare_action(self.config, batch_dict)
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align:
            batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )


class CustomizedRobotwinDataset(Dataset):
    def __init__(
        self,
        repo_id="robotwin",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        self.normalizer = Normalizer(
            # norm_stats=self.dataset.meta.stats,
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='customized',
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (normalized_item["observation.images.cam_left_wrist"] * 255).to(
            torch.uint8
        )
        right_wrist_image = (normalized_item["observation.images.cam_right_wrist"] * 255).to(
            torch.uint8
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )