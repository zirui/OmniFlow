from __future__ import annotations

"""
TorchTitan experiment: wan_new2

Design:
- Keep the experiment structure similar to `src/wan_torchtitan/`
- Use `Wan2.x` "official-like" native PyTorch modules from `omniflow.models.wan_new2`
- Do VAE/T5 preprocessing in a custom Trainer (like flux), and keep the model as the DiT backbone only
"""

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.protocols.train_spec import TrainSpec

from .adamw_fp32_state import build_wan_new2_optimizers
from .loss import build_wan_new2_loss
from .parallelize import parallelize_wan_new2
from .wan_args import WanNew2ModelArgs
from .wan_dataset import build_wan_dataloader
from .wan_model import WanNew2DiTModel


__all__ = [
    "WanNew2ModelArgs",
    "WanNew2DiTModel",
    "parallelize_wan_new2",
    "wan_new2_args",
    "get_train_spec",
]


# Model flavors
wan_new2_args = {
    "default": WanNew2ModelArgs(),
    "wan2.1_t2v_debug": WanNew2ModelArgs(
        model_type="t2v",
        dit_hidden_size=1536,
        dit_num_layers=10,
        dit_num_heads=12,
        dit_intermediate_size=8960,
        dit_in_channels=16,
        dit_out_channels=16,
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
    "wan2.1_t2v_1.3b": WanNew2ModelArgs(
        model_type="t2v",
        dit_hidden_size=1536,
        dit_num_layers=30,
        dit_num_heads=12,
        dit_intermediate_size=8960,
        dit_in_channels=16,
        dit_out_channels=16,
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
    "wan2.1_t2v_14b": WanNew2ModelArgs(
        model_type="t2v",
        dit_hidden_size=5120,
        dit_num_layers=48,
        dit_num_heads=40,
        dit_intermediate_size=13824,
        dit_in_channels=16,
        dit_out_channels=16,
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
    # Wan2.2 uses z_dim=48
    "wan2.2_t2v_debug": WanNew2ModelArgs(
        model_type="t2v",
        dit_hidden_size=3072,
        dit_num_layers=10,
        dit_num_heads=24,
        dit_intermediate_size=14336,
        dit_in_channels=48,
        dit_out_channels=48,
        vae_type="wan2.2",
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
    "wan2.2_t2v_5b": WanNew2ModelArgs(
        model_type="t2v",
        dit_hidden_size=3072,
        dit_num_layers=30,
        dit_num_heads=24,
        dit_intermediate_size=14336,
        dit_in_channels=48,
        dit_out_channels=48,
        vae_type="wan2.2",
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=WanNew2DiTModel,
        model_args=wan_new2_args,
        parallelize_fn=parallelize_wan_new2,
        pipelining_fn=None,
        build_optimizers_fn=build_wan_new2_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_wan_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=build_wan_new2_loss,
        state_dict_adapter=None,
    )