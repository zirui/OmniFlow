from dataclasses import dataclass, field
from typing import List

from torchtitan.protocols.model import BaseModelArgs


@dataclass
class WanModelArgs(BaseModelArgs):
    # DiT model parameters
    dit_hidden_size: int = 3072
    dit_num_layers: int = 30
    dit_num_heads: int = 24
    dit_intermediate_size: int = 14336
    dit_patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    dit_in_channels: int = 48
    dit_out_channels: int = 48

    dit_freq_dim: int = 256
    dit_text_dim: int = 4096
    dit_eps: float = 1.0e-6
    dit_has_image_input: bool = False
    dit_has_image_pos_emb: bool = False
    dit_has_ref_conv: bool = False
    dit_add_control_adapter: bool = False
    dit_in_channels_control_adapter: int = 24
    seperated_timestep: bool = True
    require_clip_embedding: bool = False
    require_vae_embedding: bool = False
    fuse_vae_embedding_in_latents: bool = True
    trainable_modules: str = "dit"
    t5_checkpoint_path: str = None
    vae_checkpoint_path: str = None
    vae_type: str = "wan_video_vae_38"
    mixed_precision_param: str = "float32"  # can be "float32" or "bfloat16"

    def update_from_config(self, job_config, **kwargs) -> None:
        # Update args from job_config if needed.
        if hasattr(job_config, "encoder"):
            self.mixed_precision_param = job_config.encoder.mixed_precision_param if hasattr(job_config.encoder, 'mixed_precision_param') else self.mixed_precision_param
            self.vae_checkpoint_path = job_config.encoder.vae_checkpoint_path
            self.vae_type = job_config.encoder.vae_type

    def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, float]:
        # Calculate params
        nparams = sum(p.numel() for p in model.parameters())
        # Approximate FLOPS (dummy for now)
        flops_per_token = 1.0

        # TODO: zirui, Calculate memory, for debug
        mem = sum(p.numel() * p.element_size() for p in model.parameters())
        mem_buffer = sum(b.numel() * b.element_size() for b in model.buffers())
        print("wan mem", (mem + mem_buffer) / 1024**2, "MB")

        return nparams, flops_per_token
