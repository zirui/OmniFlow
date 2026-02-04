from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from torchtitan.protocols.model import BaseModelArgs


@dataclass
class WanNew2ModelArgs(BaseModelArgs):
    """
    Model args for the vendored official-like Wan2.x DiT (`omniflow.models.wan_new2.wan_dit.WanModel`).

    Keep the naming consistent with `wan_torchtitan.WanModelArgs` as much as possible,
    but add a few knobs required by the official module.
    """

    # Task type: t2v / i2v / ti2v / s2v
    model_type: str = "t2v"

    # DiT model parameters
    text_len: int = 512
    dit_hidden_size: int = 1536
    dit_num_layers: int = 30
    dit_num_heads: int = 12
    dit_intermediate_size: int = 8960
    dit_patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    dit_in_channels: int = 16
    dit_out_channels: int = 16

    dit_freq_dim: int = 256
    dit_text_dim: int = 4096
    dit_eps: float = 1.0e-6

    # Official Wan DiT extra knobs (optional)
    dit_window_size: Tuple[int, int] = (-1, -1)
    dit_qk_norm: bool = True
    dit_cross_attn_norm: bool = True

    # OmniFlow compatibility flags
    seperated_timestep: bool = True
    fuse_vae_embedding_in_latents: bool = True
    trainable_modules: str = "dit"

    # Encoder paths
    t5_checkpoint_path: str | None = None
    vae_checkpoint_path: str | None = None
    # "wan2.1" -> z_dim=16, "wan2.2" -> z_dim=48
    vae_type: str = "wan2.1"

    def update_from_config(self, job_config, **kwargs) -> None:
        # Keep the pattern used by `wan_torchtitan`: pull encoder paths from job_config.encoder
        if hasattr(job_config, "encoder"):
            self.vae_checkpoint_path = getattr(job_config.encoder, "vae_checkpoint_path", self.vae_checkpoint_path)
            self.t5_checkpoint_path = getattr(job_config.encoder, "t5_checkpoint_path", self.t5_checkpoint_path)
            self.vae_type = getattr(job_config.encoder, "vae_type", self.vae_type)

        # Allow model_type override through custom config if present
        if hasattr(job_config, "wan"):
            self.model_type = getattr(job_config.wan, "model_type", self.model_type)

    def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, float]:
        # Rough stats for logging; we don't try to compute exact FLOPs yet.
        nparams = sum(p.numel() for p in model.parameters())
        flops_per_token = 1.0
        return nparams, flops_per_token

