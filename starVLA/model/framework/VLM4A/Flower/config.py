# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""FlowerVLA configuration — default config and FLOWER-to-starVLA conversion."""

from dataclasses import dataclass, field


@dataclass
class FlowerVLADefaultConfig:
    """FlowerVLA framework default parameters.

    Integrates the FLOWER VLA architecture (Florence-2 + Rectified Flow DiT)
    into starVLA. All fields can be overridden by the corresponding key in
    the YAML ``framework:`` section.
    """

    name: str = "FlowerVLA"

    # Legacy compat: required by some starVLA tooling
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "microsoft/Florence-2-base",
            "vl_hidden_dim": 768,
        }
    )

    # Action model settings (legacy compat)
    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 7,
            "action_horizon": 10,
            "num_inference_timesteps": 4,
        }
    )

    # FLOWER-specific architecture config
    flower: dict = field(
        default_factory=lambda: {
            # --- VLM Configuration ---
            "vlm_path": "microsoft/Florence-2-base",
            "freeze_florence": False,
            "freeze_vision_tower": False,
            "vlm_prompt_style": "default",
            "token_dropout": 0.2,
            # --- Model Structure ---
            "multistep": 10,
            "num_sampling_steps": 4,
            "lowdim_obs_dim": 7,
            "action_dim": 7,
            "act_window_size": 10,
            # --- Model Flags ---
            "use_second_view": True,
            "second_view_key": "rgb_gripper",
            "action_type_adaln": True,
            "use_causal_attention": True,
            "use_cross_attn": True,
            "use_adaln_cond": False,
            "use_readout_token": False,
            "use_proprio": False,
            "return_act_chunk": False,
            # --- DiT Configuration ---
            "sampling_type": "ln",
            "dit_dim": 512,
            "n_heads": 16,
            "n_layers": 12,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "mlp_pdrop": 0.1,
            # --- RoPE Configuration ---
            "use_rope": False,
            "use_nope": False,
            "query_seq_len": 128,
            "rope_theta": 32.0,
            # --- Checkpoint loading ---
            "load_pretrained": False,
            "pretrained_model_path": None,
        }
    )


def _build_config_from_flower_hparams(hparams: dict) -> dict:
    """Build a starVLA-compatible config dict from FLOWER checkpoint hyperparameters."""
    flower_cfg = {
        "vlm_path": hparams.get("vlm_path", "microsoft/Florence-2-base"),
        "freeze_florence": hparams.get("freeze_florence", False),
        "freeze_vision_tower": hparams.get("freeze_vision_tower", False),
        "vlm_prompt_style": hparams.get("vlm_prompt_style", "default"),
        "token_dropout": hparams.get("token_dropout", 0.2),
        "multistep": hparams.get("multistep", 10),
        "num_sampling_steps": hparams.get("num_sampling_steps", 4),
        "lowdim_obs_dim": hparams.get("lowdim_obs_dim", 7),
        "action_dim": hparams.get("action_dim", 7),
        "act_window_size": hparams.get("act_window_size", 10),
        "use_second_view": hparams.get("use_second_view", False),
        "second_view_key": hparams.get("second_view_key", "rgb_gripper"),
        "action_type_adaln": hparams.get("action_type_adaln", True),
        "use_causal_attention": hparams.get("use_causal_attention", True),
        "use_cross_attn": hparams.get("use_cross_attn", True),
        "use_adaln_cond": hparams.get("use_adaln_cond", False),
        "use_readout_token": hparams.get("use_readout_token", False),
        "use_proprio": hparams.get("use_proprio", False),
        "sampling_type": hparams.get("sampling_type", "ln"),
        "dit_dim": hparams.get("dit_dim", 512),
        "n_heads": hparams.get("n_heads", 16),
        "n_layers": hparams.get("n_layers", 12),
        "attn_pdrop": hparams.get("attn_pdrop", 0.1),
        "resid_pdrop": hparams.get("resid_pdrop", 0.1),
        "mlp_pdrop": hparams.get("mlp_pdrop", 0.1),
        "use_rope": hparams.get("use_rope", False),
        "use_nope": hparams.get("use_nope", False),
        "query_seq_len": hparams.get("query_seq_len", 128),
        "rope_theta": hparams.get("rope_theta", 32.0),
        "load_pretrained": hparams.get("load_pretrained", False),
        "pretrained_model_path": hparams.get("pretrained_model_path", None),
    }

    return {
        "framework": {
            "name": "FlowerVLA",
            "qwenvl": {
                "base_vlm": flower_cfg["vlm_path"],
                "vl_hidden_dim": 768 if "base" in flower_cfg["vlm_path"] else 1024,
            },
            "action_model": {
                "action_dim": flower_cfg["action_dim"],
                "action_horizon": flower_cfg["act_window_size"],
                "num_inference_timesteps": flower_cfg["num_sampling_steps"],
            },
            "flower": flower_cfg,
        },
        "version_id": "0.21",
    }
