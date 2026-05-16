# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Data bridge — convert starVLA ``List[dict]`` format ↔ FLOWER batch dict."""

from typing import Dict, List

import numpy as np
import torch
from PIL import Image

DEFAULT_IMAGE_SIZE = (200, 200)

# CLIP normalization stats used by FLOWER training (OpenCLIP variant)
FLOWER_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FLOWER_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Apply CLIP normalization (scale [0,1] and normalize with ImageNet stats)."""
    mean = torch.tensor(FLOWER_IMAGE_MEAN, device=t.device, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(FLOWER_IMAGE_STD, device=t.device, dtype=t.dtype).view(3, 1, 1)
    return (t - mean) / std


def _pil_list_to_image_tensor(images: List[Image.Image], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert a list of PIL images to a batched tensor (B, T, C, H, W) with T=1."""
    batch_tensors = []
    for img in images:
        if isinstance(img, Image.Image):
            arr = np.array(img)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        elif isinstance(img, np.ndarray):
            t = torch.from_numpy(img).float()
            if t.ndim == 3 and t.shape[-1] in (3, 4):
                t = t.permute(2, 0, 1)
            if t.max() > 1.0:
                t = t / 255.0
        elif isinstance(img, torch.Tensor):
            t = img.float()
            if t.ndim == 3 and t.shape[-1] in (3, 4):
                t = t.permute(2, 0, 1)
            if t.max() > 1.0:
                t = t / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        t = _normalize_tensor(t)
        # Add time dimension: (C, H, W) → (1, C, H, W)
        t = t.unsqueeze(0)
        batch_tensors.append(t)
    # Stack to (B, T, C, H, W)
    return torch.stack(batch_tensors, dim=0).to(device=device, dtype=dtype)


def _starVLA_to_flower_batch(
    examples: List[dict],
    device: torch.device,
    dtype: torch.dtype,
    action_horizon: int = 10,
    include_actions: bool = True,
) -> Dict:
    """
    Bridge starVLA's List[dict] format to FLOWER's batch dict format.

    Args:
        examples: List[dict], each with keys:
            - image: List[PIL.Image]  (rgb_static, rgb_gripper for CALVIN)
            - lang: str
            - action: np.ndarray [T, action_dim] (required if include_actions=True)
        device: target device
        dtype: target dtype
        action_horizon: number of action steps to take from the end
        include_actions: whether to include actions (False for inference)

    Returns:
        dict with keys: rgb_obs, lang_text, actions (optional)
    """
    batch_size = len(examples)

    # Extract images: examples[i]["image"] = [rgb_static_PIL, rgb_gripper_PIL]
    rgb_static_list = []
    rgb_gripper_list = []
    for ex in examples:
        imgs = ex["image"]
        rgb_static_list.append(imgs[0])
        rgb_gripper_list.append(imgs[1] if len(imgs) > 1 else imgs[0])

    rgb_static = _pil_list_to_image_tensor(rgb_static_list, device, dtype)
    rgb_gripper = _pil_list_to_image_tensor(rgb_gripper_list, device, dtype)

    # Extract language instructions
    lang_text = [ex["lang"] for ex in examples]

    batch = {
        "rgb_obs": {
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
        },
        "lang_text": lang_text,
    }

    if include_actions:
        actions_list = []
        for ex in examples:
            act = ex["action"]
            if isinstance(act, np.ndarray):
                act = torch.from_numpy(act).to(device=device, dtype=dtype)
            elif isinstance(act, list):
                act = torch.tensor(act, device=device, dtype=dtype)
            # Take the last action_horizon steps
            act = act[-action_horizon:, :]
            actions_list.append(act)
        batch["actions"] = torch.stack(actions_list, dim=0)

    return batch
