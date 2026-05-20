# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Checkpoint loading mixin — key remapping, state dict loading, from_pretrained."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

from starVLA.model.framework.VLM4A.Flower.config import _build_config_from_flower_hparams
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class FlowerVLACheckpointMixin:
    """Mixin providing checkpoint loading with FLOWER key remapping.

    Must appear *before* ``baseframework`` in the MRO so that
    ``load_state_dict`` intercepts calls before ``nn.Module``.
    """

    # ── Key remapping ────────────────────────────────────────────────

    @staticmethod
    def _remap_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remap FLOWER checkpoint keys to match starVLA model structure.

        Both ``_load_pretrained_weights`` and the starVLA pipeline's
        ``baseframework.from_pretrained`` path call ``load_state_dict``,
        so this hook ensures key remapping always happens regardless of
        the entry point.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            new_key = new_key.replace("agent.", "")

            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")

            if ".language_shared." in new_key:
                new_key = new_key.replace("vlm.language_shared.", "vlm.language_model.model.shared.")
            if ".language_final_logits_bias" in new_key:
                new_key = new_key.replace("vlm.language_final_logits_bias", "vlm.language_model.final_logits_bias")

            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state_dict[new_key] = value
        return new_state_dict

    # ── State dict loading ───────────────────────────────────────────

    # Keys that are safe to skip because they are deterministically
    # initialised (buffers) or tied to another weight in the checkpoint.
    _FLOWER_SAFE_MISSING_KEYS = {
        # buffer computed from fixed sinusoidal formula; identical every init
        "vlm.visual_temporal_embed.pos_idx_to_embed",
        # zero-initialised buffer registered by Florence2ForConditionalGeneration
        "vlm.language_model.final_logits_bias",
        # tied to vlm.language_model.model.shared.weight which IS in the ckpt
        "vlm.language_model.model.encoder.embed_tokens.weight",
    }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Overridden to remap FLOWER checkpoint keys before loading."""
        remapped = self._remap_state_dict_keys(state_dict)

        # Patch tied embedding so that strict=True passes when the ckpt
        # stores ``shared.weight`` but the model also expects ``embed_tokens.weight``.
        shared_key = "vlm.language_model.model.shared.weight"
        embed_key = "vlm.language_model.model.encoder.embed_tokens.weight"
        if shared_key in remapped and embed_key not in remapped:
            remapped[embed_key] = remapped[shared_key]

        if strict:
            model_keys = set(super().state_dict().keys())
            ckpt_keys = set(remapped.keys())
            missing = model_keys - ckpt_keys
            safe_missing = missing & self._FLOWER_SAFE_MISSING_KEYS
            if safe_missing:
                # Fill safe-missing keys from the model's own current values
                # (they are deterministic buffers / tied weights).
                own_state = super().state_dict()
                for k in safe_missing:
                    remapped[k] = own_state[k]
                logger.info(
                    "Patched %d safe-missing key(s) into state dict: %s",
                    len(safe_missing),
                    sorted(safe_missing),
                )

        return super().load_state_dict(remapped, strict=strict, assign=assign)

    # ── Weight loading with EMA handling ─────────────────────────────

    def _load_pretrained_weights(self, pretrained_model_path: str):
        """Load pretrained FLOWER weights with EMA handling.

        Key remapping is handled automatically by ``load_state_dict``.
        """
        logger.info(f"Loading pretrained weights from {pretrained_model_path}...")

        if pretrained_model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(pretrained_model_path, device="cpu")
            checkpoint = {"state_dict": state_dict}
        else:
            checkpoint = torch.load(pretrained_model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

        # Handle EMA weights (PyTorch Lightning EMA callback format)
        if (
            "callbacks" in checkpoint
            and "EMA" in checkpoint.get("callbacks", {})
            and "ema_weights" in checkpoint["callbacks"]["EMA"]
        ):
            logger.info("Found EMA weights in checkpoint, extracting them...")
            ema_weights_list = checkpoint["callbacks"]["EMA"]["ema_weights"]
            original_state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {}
            ema_idx = 0

            for param_name, original_param in original_state_dict.items():
                if ema_idx < len(ema_weights_list):
                    ema_weight = ema_weights_list[ema_idx]
                    if ema_weight.shape == original_param.shape:
                        state_dict[param_name] = ema_weight
                        ema_idx += 1
                    else:
                        found = False
                        for temp_idx in range(ema_idx, min(ema_idx + 20, len(ema_weights_list))):
                            if ema_weights_list[temp_idx].shape == original_param.shape:
                                state_dict[param_name] = ema_weights_list[temp_idx]
                                ema_weights_list[temp_idx], ema_weights_list[ema_idx] = (
                                    ema_weights_list[ema_idx], ema_weights_list[temp_idx]
                                )
                                ema_idx += 1
                                found = True
                                break
                        if not found:
                            state_dict[param_name] = original_param
                else:
                    state_dict[param_name] = original_param

            logger.info(f"Matched {ema_idx} EMA weights out of {len(ema_weights_list)} total")

        # Key remapping happens inside load_state_dict
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:20]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:20]}...")
        if not missing_keys and not unexpected_keys:
            logger.info("All keys matched successfully!")

        return missing_keys, unexpected_keys

    # ── from_pretrained classmethod ──────────────────────────────────

    @classmethod
    def from_pretrained(cls, pretrained_checkpoint: str, **kwargs):
        """Load a FlowerVLA model from a checkpoint.

        Supports:
          1. starVLA mode: starVLA-style config.yaml in checkpoint dir
          2. FLOWER directory mode: dir with model.safetensors + config.yaml
          3. FLOWER ckpt mode: .ckpt/.pt with hyper_parameters embedded

        Args:
            pretrained_checkpoint: Path to checkpoint file (.ckpt, .pt, .safetensors)
                or a directory containing model.safetensors + config.yaml.
            **kwargs: Passed through to base class

        Returns:
            FlowerVLA: Model with loaded weights
        """
        ckpt_path = Path(pretrained_checkpoint)

        # Resolve directory: if a file is given, its parent dir may contain config.yaml
        if ckpt_path.is_dir():
            ckpt_dir = ckpt_path
            # Find the weight file in the directory
            weight_files = list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.ckpt")) + list(ckpt_dir.glob("*.pt"))
            if not weight_files:
                raise FileNotFoundError(f"No weight file (*.safetensors, .ckpt, .pt) found in {ckpt_dir}")
            weight_path = weight_files[0]
        else:
            weight_path = ckpt_path
            ckpt_dir = ckpt_path.parent

        # ── Resolve model config ──
        norm_stats = {"action": {"mean": 0.0, "std": 1.0}}

        # Try starVLA config first
        flower_config = ckpt_dir / "config.yaml"

        try:
            model_config, norm_stats = read_mode_config(ckpt_dir)
            config = dict_to_namespace(model_config)
            logger.info("Loaded starVLA config from %s", ckpt_dir)
        except (FileNotFoundError, AssertionError):
            # FLOWER mode: load hparams from companion config.yaml or checkpoint
            config = None
            if flower_config.exists():
                try:
                    from omegaconf import OmegaConf

                    flower_cfg = OmegaConf.load(flower_config)
                    flower_model = OmegaConf.to_container(flower_cfg.get("model", {}), resolve=True)
                    if flower_model:
                        hparams = {k: v for k, v in flower_model.items() if not k.startswith("_")}
                        # The config.yaml uses _target_, _recursive_ etc from Hydra — filter them
                        clean_hparams = {}
                        for k, v in hparams.items():
                            if isinstance(v, dict) and "_target_" in v:
                                continue  # skip hydra-instantiated sub-objects
                            clean_hparams[k] = v
                        # Map model config keys to flower cfg keys
                        model_config = _build_config_from_flower_hparams(clean_hparams)
                        config = dict_to_namespace(model_config)
                        logger.info("Built config from FLOWER config.yaml")
                except Exception as e:
                    logger.warning("Failed to parse FLOWER config.yaml: %s", e)

            if config is None:
                # Last resort: extract hparams from checkpoint
                logger.info("Extracting hparams from FLOWER checkpoint...")
                if weight_path.suffix == ".safetensors":
                    from safetensors.torch import load_file

                    state_dict = load_file(str(weight_path), device="cpu")
                    checkpoint = {"state_dict": state_dict}
                else:
                    checkpoint = torch.load(str(weight_path), map_location="cpu")
                hparams = checkpoint.get("hyper_parameters", checkpoint.get("state_dict", {}).get("hyper_parameters", {}))
                model_config = _build_config_from_flower_hparams(hparams)
                config = dict_to_namespace(model_config)

        # Ensure framework name is correct and disable recursive pretrained loading
        # (the .safetensors we are about to load already contains the full weights)
        from omegaconf import OmegaConf

        OmegaConf.update(config, "framework.name", "FlowerVLA", force_add=True)
        OmegaConf.update(config, "trainer.pretrained_checkpoint", None, force_add=True)
        OmegaConf.update(config, "framework.flower.load_pretrained", False, force_add=True)

        # Build model
        from starVLA.model.framework.base_framework import build_framework

        model = build_framework(cfg=config)
        model.norm_stats = norm_stats

        # Load FLOWER weights
        model._load_pretrained_weights(str(weight_path))

        return model
