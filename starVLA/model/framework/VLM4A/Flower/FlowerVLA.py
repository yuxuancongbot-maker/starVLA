# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
FlowerVLA Framework — FLOWER Vision-Language-Action model for starVLA.

Integrates the FLOWER VLA architecture (Florence-2 backbone + Rectified Flow DiT)
into starVLA's modular framework system.

Architecture:
  Florence-2 DaVit Vision Encoder → LLM Encoder Layers → Cross-Attention DiT
  → AdaLN Conditioning → Action Decoder (Rectified Flow, Euler integration)

Key features:
  - ~1B parameters, <3GB VRAM inference
  - SOTA on CALVIN ABC→D (4.53 Avg Length) and LIBERO benchmarks
  - 4 denoising steps for fast inference
  - 10-step action chunking

Reference: https://github.com/intuitive-robots/flower_vla_calvin
"""

import functools
import logging
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.layers.mlp import Mlp

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.Flower.checkpoint import FlowerVLACheckpointMixin
from starVLA.model.framework.VLM4A.Flower.config import FlowerVLADefaultConfig, _build_config_from_flower_hparams
from starVLA.model.framework.VLM4A.Flower.data_bridge import _starVLA_to_flower_batch
from starVLA.model.framework.VLM4A.Flower.flower_transformers import (
    ActionSpaceEmbedderParameter,
    FlowBlock,
    FreqEmbedder,
    RmsNorm,
    SharedAdaLNController,
    TimestepEmbedder,
    ZeroEncoder,
    stateless_norm,
)
from starVLA.model.framework.VLM4A.Flower.flower_utils import ActionIndex, generate_policy_prompt
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Data bridge and config utilities moved to data_bridge.py / config.py / checkpoint.py


# _build_config_from_flower_hparams → config.py
# FlowerVLADefaultConfig → config.py


@FRAMEWORK_REGISTRY.register("FlowerVLA")
class FlowerVLA(FlowerVLACheckpointMixin, baseframework):
    """
    FLOWER Vision-Language-Action model for starVLA.

    Components:
      - Florence-2 VLM backbone (DaVit vision encoder + LLM encoder layers)
      - Rectified Flow DiT with cross-attention and AdaLN conditioning
      - Action chunking (multistep) for multi-step action generation

    The model predicts delta end-effector actions (7-dim: xyz+rot+gripper)
    from visual observations (rgb_static + rgb_gripper) and language instructions.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(FlowerVLADefaultConfig, config)

        # Extract FLOWER-specific config
        flower_cfg = self.config.framework.flower
        self._init_from_flower_cfg(flower_cfg)

    def _init_from_flower_cfg(self, flower_cfg: dict) -> None:
        """Initialize all model components from a FLOWER config dict."""
        # ── Resolve config values with defaults ──
        vlm_path = flower_cfg.get("vlm_path", "microsoft/Florence-2-base")
        freeze_florence = flower_cfg.get("freeze_florence", False)
        freeze_vision_tower = flower_cfg.get("freeze_vision_tower", False)
        vlm_prompt_style = flower_cfg.get("vlm_prompt_style", "default")
        token_dropout = flower_cfg.get("token_dropout", 0.2)

        self.multistep = int(flower_cfg.get("multistep", 10))
        self.num_sampling_steps = int(flower_cfg.get("num_sampling_steps", 4))
        lowdim_obs_dim = int(flower_cfg.get("lowdim_obs_dim", 7))
        self.action_dim = int(flower_cfg.get("action_dim", 7))
        self.act_window_size = int(flower_cfg.get("act_window_size", 10))

        self.use_second_view = bool(flower_cfg.get("use_second_view", True))
        self.second_view_key = flower_cfg.get("second_view_key", "rgb_gripper")
        self.use_cross_attn = bool(flower_cfg.get("use_cross_attn", True))
        self.use_rope = bool(flower_cfg.get("use_rope", False))
        self.use_nope = bool(flower_cfg.get("use_nope", False))
        self.use_adaln_cond = bool(flower_cfg.get("use_adaln_cond", False))
        self.use_readout_token = bool(flower_cfg.get("use_readout_token", False)) and self.use_adaln_cond
        self.action_type_adaln = bool(flower_cfg.get("action_type_adaln", True))
        self.use_proprio = bool(flower_cfg.get("use_proprio", False))
        self.return_act_chunk = bool(flower_cfg.get("return_act_chunk", False))

        sampling_type = flower_cfg.get("sampling_type", "ln")
        if sampling_type not in ("ln", "pi_zero", "uniform", "stratified", "loglogistic"):
            raise ValueError(f"Invalid sampling type: {sampling_type}")
        self.sampling_type = sampling_type

        dit_dim = int(flower_cfg.get("dit_dim", 512))
        n_heads = int(flower_cfg.get("n_heads", 16))
        n_layers = int(flower_cfg.get("n_layers", 12))
        attn_pdrop = float(flower_cfg.get("attn_pdrop", 0.1))
        resid_pdrop = float(flower_cfg.get("resid_pdrop", 0.1))
        mlp_pdrop = float(flower_cfg.get("mlp_pdrop", 0.1))
        query_seq_len = int(flower_cfg.get("query_seq_len", 128))
        rope_theta = float(flower_cfg.get("rope_theta", 32.0))
        self.dit_dim = dit_dim

        self.vlm_prompt_style = vlm_prompt_style
        self.token_dropout = token_dropout

        # ── Setup prompt formatter ──
        self.format_instruction = functools.partial(
            generate_policy_prompt,
            robot_name="Franka Panda",
            action_space="Delta End-Effector",
            num_arms="1",
            prompt_style="minimal",
        )

        # ── Action space index ──
        self.action_space_index = ActionIndex()

        # ── Load Florence-2 VLM ──
        logger.info(f"Loading Florence-2 VLM from {vlm_path}...")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Monkey-patch: Florence2PreTrainedModel defines _supports_sdpa as a
        # @property that delegates to self.language_model._supports_sdpa, but
        # during __init__ the language_model doesn't exist yet, so the property
        # raises AttributeError.  The architecture DOES support SDPA (it has
        # Florence2SdpaAttention), so we set the class attr directly before
        # loading so the init-time check passes.
        sdpa_patched = False
        try:
            config = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True)
            class_ref = config.auto_map["AutoModelForCausalLM"]
            model_cls = get_class_from_dynamic_module(
                class_ref, vlm_path, trust_remote_code=True
            )
            model_cls._supports_sdpa = True
            sdpa_patched = True
        except Exception:
            logger.warning("Failed to enable SDPA on Florence-2 class; falling back to eager.")

        vlm_kwargs = {"trust_remote_code": True}
        if not sdpa_patched:
            vlm_kwargs["attn_implementation"] = "eager"
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, **vlm_kwargs)

        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # ── Create prompt embedding (must be BEFORE deleting decoder;
        # resize_token_embeddings needs decoder.embed_tokens) ──
        self.prompt_embeds = self._create_prompt_embed("<Flow>")

        # Delete decoder and lm_head (not needed for encoding)
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head

        # Get hidden dimension from VLM config
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim

        # Update qwenvl config for compat
        if hasattr(self.config, "framework") and hasattr(self.config.framework, "qwenvl"):
            self.config.framework.qwenvl.vl_hidden_dim = hidden_dim

        # ── Token dropout ──
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

        # ── Build DiT components ──
        # Core conditioning components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            dit_dim, max_actions=len(self.action_space_index.action_spaces)
        )

        # Positional encoding
        if not self.use_rope and not self.use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, self.act_window_size, dit_dim) * 0.1
            )

        # Action encoders/decoders
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()

        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim, hidden_features=dit_dim,
                out_features=dit_dim, bias=True
            )
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim)

            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    dit_dim, global_conddim=dit_dim, use_cross_attn=self.use_cross_attn
                )

            if self.use_proprio:
                self.proprio_encoders[action_name] = (
                    Mlp(input_dim, dit_dim, out_features=dit_dim, drop=0.2)
                    if action_name == "bimanual_nav"
                    else ZeroEncoder(self.dit_dim, device=None)
                )

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock(
                dit_dim, n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_pdrop=mlp_pdrop,
                use_cross_attn=self.use_cross_attn,
                use_rope=self.use_rope,
                query_seq_len=query_seq_len,
                rope_theta=rope_theta,
            ) for _ in range(n_layers)
        ])

        # ── Rollout state ──
        self.rollout_step_counter = 0
        self.pred_action_seq = None

        # ── Optionally load pretrained weights ──
        load_pretrained = flower_cfg.get("load_pretrained", False)
        pretrained_path = flower_cfg.get("pretrained_model_path", None)
        if load_pretrained and pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    # ══════════════════════════════════════════════════════════════════
    # Prompt / Text Embedding helpers
    # ══════════════════════════════════════════════════════════════════

    def _create_prompt_embed(self, prompt_text: str) -> nn.Parameter:
        """Create learnable embedding for a special prompt token."""
        self.tokenizer.add_special_tokens({"additional_special_tokens": [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False,
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def construct_prompts(self, dataset_batch: Dict) -> List[str]:
        """Construct formatted prompts for Florence-2 encoder conditioning."""
        language_instruction = dataset_batch["lang_text"]
        text_prompts = []
        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                text_prompts.append(self.format_instruction(instruction))
            elif self.vlm_prompt_style == "feature_focused":
                prompt = f"<od>{instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                text_prompts.append(prompt)
            elif self.vlm_prompt_style == "state_oriented":
                prompt = f"<od>{instruction}</od><referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                text_prompts.append(prompt)
            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")
        return text_prompts

    def _get_text_embeddings(self, text: List[str], device: torch.device) -> torch.Tensor:
        """Tokenize text and return input embeddings."""
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])

    # ══════════════════════════════════════════════════════════════════
    # Observation Encoding
    # ══════════════════════════════════════════════════════════════════

    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Encode visual observations and language through Florence-2 VLM.

        Returns a dict with keys: features, frequency_embeds, action_space_embeds,
        action_type, proprio, attention_mask.
        """
        device = self.vlm.device
        default_type = next(self.vlm.parameters()).dtype

        embed_tensor = torch.zeros(len(batch["rgb_obs"]["rgb_static"]), 1, 1)
        action_type_tensor = torch.ones(
            len(batch["rgb_obs"]["rgb_static"]), self.act_window_size, 7
        )

        # Process primary image (rgb_static)
        image_tensor = batch["rgb_obs"]["rgb_static"]
        B, T, C, H, W = image_tensor.shape

        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        # Process second view (rgb_gripper) if enabled
        if self.use_second_view:
            image2_tensor = batch["rgb_obs"].get("rgb_gripper", None)
            if image2_tensor is not None:
                _, _, C2, H2, W2 = image2_tensor.shape
                image2_features = self.vlm._encode_image(
                    image2_tensor.view(-1, C2, H2, W2).to(device).to(default_type)
                ).to(default_type)
                image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
                image_features = torch.cat([image_features, image2_features], dim=1)

        # Get text embeddings
        constructed_prompts = self.construct_prompts(batch)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)

        # Add task prompt token
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)

        # Merge sequence: [image_features, task_prompt, text_embeds]
        merged_embeds = torch.cat([
            image_features,
            task_prompt,
            text_embeds.to(image_features.device),
        ], dim=1)

        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)

        # Run through LLM encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        # Apply token dropout
        features = self.vlm_token_dropout(features)

        # Frequency embedding (fixed at index 3)
        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor).to(device) * 3
        )

        # Proprioception
        proprio = None
        if self.use_proprio and "proprio" in batch:
            proprio = batch["proprio"].to(device).to(default_type)

        return {
            "features": features,
            "frequency_embeds": frequency_embeds,
            "action_space_embeds": None,
            "action_type": torch.ones_like(action_type_tensor),
            "proprio": proprio,
            "attention_mask": attention_mask,
        }

    # ══════════════════════════════════════════════════════════════════
    # Action Encoding / Decoding
    # ══════════════════════════════════════════════════════════════════

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode actions using action-specific encoders."""
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.vlm.device)
        batch_size = z.shape[0]
        encoded = torch.zeros(batch_size, z.shape[1], self.dit_dim, device=self.vlm.device).to(default_dtype)
        valid_dims = torch.zeros_like(z).to(default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.action_encoders[action_name](z)
        return encoded, valid_dims

    def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        """Decode actions using action-specific decoders."""
        default_dtype = next(self.parameters()).dtype
        batch_size = z.shape[0]
        decoded = torch.zeros(batch_size, z.shape[1], self.action_dim,
                             device=self.vlm.device).to(default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                pred = self.action_decoders[action_name](z)
                decoded = pred
        return decoded

    # ══════════════════════════════════════════════════════════════════
    # Rectified Flow Loss & Sampling
    # ══════════════════════════════════════════════════════════════════

    def rf_loss(self, cond: Dict, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute rectified flow loss: E_t [ || (z1 - a) - v_theta(z_t, t) ||^2 ]."""
        default_dtype = next(self.parameters()).dtype

        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        # Sample time
        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        # Interpolate: z_t = (1-t)*actions + t*noise
        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.randn_like(actions, device=device).to(default_dtype)
        zt = (1 - texp) * actions + texp * z1

        # Predict velocity field
        vtheta = self.dit_forward(zt, t, cond)

        # L2 loss
        diff = (z1 - actions) - vtheta
        loss = (diff ** 2).mean()

        losses_dict = {
            "diff_min": diff.min().item(),
            "diff_max": diff.max().item(),
            "diff_mean": diff.mean().item(),
            "loss": loss.item(),
        }
        return loss, losses_dict

    @torch.no_grad()
    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """Sample actions via Euler integration (rectified flow)."""
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device

        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1] * (z.dim() - 1))

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)
            vc = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc

        return z.clamp(-1, 1)

    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: Dict) -> torch.Tensor:
        """Forward pass through all DiT blocks with AdaLN conditioning."""
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape

        cond = cond_dict["features"].to(default_dtype)
        frequency_embeds = cond_dict["frequency_embeds"].squeeze(1).to(default_dtype)
        action_type = cond_dict["action_type"].to(self.vlm.device)

        # Proprioception
        if self.use_proprio and cond_dict.get("proprio") is not None:
            proprio = cond_dict["proprio"].to(default_dtype)
            proprio_embeds = self._encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)

        # Encode actions
        z, valid_dims = self.encode_actions(z, action_type)

        # Positional encoding
        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding

        # Combine timestep + frequency + proprio embeddings
        t_emb = (
            stateless_norm(self.t_embedder(t))
            + stateless_norm(frequency_embeds).squeeze(1)
            + stateless_norm(proprio_embeds).squeeze(1)
        )

        cond = self.cond_linear(self.cond_norm(cond))

        # Global conditioning
        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb

        # Context for cross-attention
        cx = z
        context = cond if self.use_cross_attn else None

        # Get AdaLN signals
        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self._action_specific_adaln(global_cond, action_type)

        # Pass through DiT blocks
        for layer in self.dit:
            cx = layer(
                cx, global_cond,
                context=context,
                is_causal=True,
                global_adaln=global_adaln,
            )

        return self.decode_actions(cx, action_type, valid_dims)

    def _encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """Encode proprioceptive state based on action type."""
        batch_size = output_shape[0]
        default_dtype = next(self.parameters()).dtype

        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.vlm.device)

        encoded_proprio = torch.zeros(batch_size, self.dit_dim, device=self.vlm.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded_proprio[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1)
        return encoded_proprio

    def _action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """Generate action-specific AdaLN modulation signals."""
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        device = global_cond.device

        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_type)
            for _ in range(num_chunks)
        ]

        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond)
                for i, signal in enumerate(action_mod):
                    mod_signals[i] = signal
        return mod_signals

    # ══════════════════════════════════════════════════════════════════
    # starVLA Framework Interface
    # ══════════════════════════════════════════════════════════════════

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image]  (rgb_static, rgb_gripper)
                - lang: str  instruction text
                - action: np.ndarray [T, action_dim]

        Returns:
            dict: {"action_loss": torch.Tensor scalar}
        """
        device = self.vlm.device
        default_dtype = next(self.vlm.parameters()).dtype

        # Bridge data format: List[dict] → FLOWER batch dict
        batch = _starVLA_to_flower_batch(
            examples, device, default_dtype,
            action_horizon=self.act_window_size, include_actions=True,
        )

        # Encode observations through Florence-2
        obs_features = self.encode_observations(batch)

        # Compute rectified flow loss
        with torch.autocast("cuda", dtype=torch.float32):
            action_loss, _ = self.rf_loss(obs_features, batch["actions"])

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> Dict[str, np.ndarray]:
        """Inference: predict future actions from observations.

        Args:
            examples: List[dict] (same schema as forward, action optional)

        Returns:
            dict: {"normalized_actions": np.ndarray [B, action_horizon, action_dim]}
        """
        if not isinstance(examples, list):
            examples = [examples]

        device = self.vlm.device
        default_dtype = next(self.vlm.parameters()).dtype

        # Bridge data format
        batch = _starVLA_to_flower_batch(
            examples, device, default_dtype,
            action_horizon=self.act_window_size, include_actions=False,
        )

        # Encode observations
        obs_features = self.encode_observations(batch)

        # Generate noise and sample actions
        with torch.autocast("cuda", dtype=torch.float32):
            noise = torch.randn(
                len(obs_features["features"]),
                self.act_window_size,
                self.action_dim,
                device=device,
                dtype=torch.float32,
            )
            pred_actions = self.sample_actions(noise, obs_features, inference=True)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    # ══════════════════════════════════════════════════════════════════
    # Step-based inference (action chunking for rollout)
    # ══════════════════════════════════════════════════════════════════

    def step(self, obs: Dict, goal: Dict) -> np.ndarray:
        """Single-step inference with action chunking.

        Compatible with starVLA's CALVIN eval pipeline which calls model.step(obs, goal).

        Args:
            obs: dict with keys "rgb_obs" (rgb_static, rgb_gripper as numpy arrays)
            goal: dict with key "lang_text" (str instruction)

        Returns:
            np.ndarray: action vector [action_dim]
        """
        if self.rollout_step_counter % self.multistep == 0:
            # Convert eval obs format to List[dict] for predict_action
            rgb_static = obs["rgb_obs"]["rgb_static"]
            rgb_gripper = obs["rgb_obs"]["rgb_gripper"]

            # Convert numpy arrays to PIL images
            if isinstance(rgb_static, np.ndarray):
                if rgb_static.dtype == np.float32 or rgb_static.dtype == np.float64:
                    rgb_static = (rgb_static * 255).astype(np.uint8)
                pil_static = Image.fromarray(rgb_static)
            else:
                pil_static = rgb_static

            if isinstance(rgb_gripper, np.ndarray):
                if rgb_gripper.dtype == np.float32 or rgb_gripper.dtype == np.float64:
                    rgb_gripper = (rgb_gripper * 255).astype(np.uint8)
                pil_gripper = Image.fromarray(rgb_gripper)
            else:
                pil_gripper = rgb_gripper

            example = {
                "image": [pil_static, pil_gripper],
                "lang": goal.get("lang_text", goal.get("lang", "")),
            }

            result = self.predict_action([example])
            self.pred_action_seq = result["normalized_actions"][0]  # (act_window_size, action_dim)

        current_action = self.pred_action_seq[self.rollout_step_counter]
        self.rollout_step_counter += 1

        if self.rollout_step_counter >= self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def reset(self):
        """Reset rollout state for a new evaluation sequence."""
        self.rollout_step_counter = 0
        self.pred_action_seq = None

    # ══════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════

    def print_model_parameters(self):
        """Print model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total Parameters: {total_params:,}")
        for name, submodule in self.named_modules():
            if "." not in name or name.count(".") <= 1:
                submodule_params = sum(p.numel() for p in submodule.parameters())
                if submodule_params > 0:
                    logger.info(f"  {name}: {submodule_params:,} params")


# ══════════════════════════════════════════════════════════════════════
# Quick smoke test (run with: python -m starVLA.model.framework.Flower.FlowerVLA)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("FlowerVLA Smoke Test")
    print("=" * 60)

    if args.checkpoint:
        # Load from checkpoint
        print(f"\nLoading from checkpoint: {args.checkpoint}")
        model = FlowerVLA.from_pretrained(args.checkpoint)
    else:
        # Build from config or defaults
        if args.config_yaml:
            cfg = OmegaConf.load(args.config_yaml)
        else:
            cfg = OmegaConf.create({"framework": {"name": "FlowerVLA"}})

        print("\nBuilding FlowerVLA from config...")
        model = FlowerVLA(cfg)

    model = model.cuda()
    model.eval()
    model.print_model_parameters()

    # Test with dummy CALVIN-style input
    print("\nTesting forward pass with dummy data...")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    dummy_gripper = Image.fromarray(np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8))
    example = {
        "image": [dummy_img, dummy_gripper],
        "lang": "pick up the red block and place it in the slider",
        "action": np.random.uniform(-1, 1, size=(20, 7)).astype(np.float32),
    }

    # Test forward
    with torch.no_grad():
        out = model([example])
        print(f"  action_loss: {out['action_loss'].item():.4f}")

    # Test predict_action
    pred = model.predict_action([example])
    actions = pred["normalized_actions"]
    print(f"  predicted_actions shape: {actions.shape}")
    print(f"  action range: [{actions.min():.3f}, {actions.max():.3f}]")

    # Test step-based inference
    model.reset()
    obs = {
        "rgb_obs": {
            "rgb_static": np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
            "rgb_gripper": np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8),
        }
    }
    goal = {"lang_text": "pick up the red block"}
    step_action = model.step(obs, goal)
    print(f"  step action shape: {step_action.shape}")

    print("\n" + "=" * 60)
    print("Smoke test completed successfully!")
    print("=" * 60)
