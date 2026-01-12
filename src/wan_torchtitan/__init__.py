from torchtitan.protocols.train_spec import TrainSpec
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.lr_scheduler import build_lr_schedulers

from .wan_args import WanModelArgs
from .wan_model import WanVideoModel
from .parallelize import parallelize_wan
from .wan_dataset import build_wan_dataloader
from .loss import build_wan_loss


# Wan arguments
wan_args = {
    "default": WanModelArgs(),
    "wan2.1_t2v_debug": WanModelArgs(
        dit_hidden_size=1536,
        dit_num_layers=10,
        dit_num_heads=12,
        dit_intermediate_size=8960,
        dit_in_channels=16,
        dit_out_channels=16,
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
    "wan2.1_t2v_1.3b": WanModelArgs(
        dit_hidden_size=1536,
        dit_num_layers=30,
        dit_num_heads=12,
        dit_intermediate_size=8960,
        dit_in_channels=16,
        dit_out_channels=16,
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
    "wan2.1_t2v_14b": WanModelArgs(
        dit_hidden_size=5120,
        dit_num_layers=48,
        dit_num_heads=40,
        dit_intermediate_size=13824,
        dit_in_channels=16,
        dit_out_channels=16,
        fuse_vae_embedding_in_latents=False,
        seperated_timestep=False,
    ),
    "wan2.2_t2v_debug": WanModelArgs(
        dit_hidden_size=3072,
        dit_num_layers=10,
        dit_num_heads=24,
        dit_intermediate_size=14336,
        dit_in_channels=48,
        dit_out_channels=48,
    ),
    "wan2.2_t2v_5b": WanModelArgs(
        dit_hidden_size=3072,
        dit_num_layers=30,
        dit_num_heads=24,
        dit_intermediate_size=14336,
        dit_in_channels=48,
        dit_out_channels=48,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=WanVideoModel,
        model_args=wan_args,
        parallelize_fn=parallelize_wan,
        pipelining_fn=None,  # Pipeline parallel not implemented yet
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_wan_dataloader,
        build_tokenizer_fn=None,  
        # Wan uses T5/TextEncoder inside.
        build_loss_fn=build_wan_loss,
        state_dict_adapter=None,  # Implement adapter for checkpointing later
    )
