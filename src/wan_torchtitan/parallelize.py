# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

def parallelize_wan(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply parallelism to the Wan model using FSDP2 (fully_shard).
    Mimics Flux parallelization strategy.
    
    Args:
        model: WanVideoModel wrapper (contains .model which is WanVideoForConditionalGeneration)
        parallel_dims: ParallelDims object
        job_config: JobConfig object
    """

    # TODO (limou)
    # enable FSDP for Wan model components
    # return model
    
    # TODO: zirui, Check if we should freeze components
    if hasattr(model, "freeze_except"):
        pass

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if parallel_dims.dp_shard_enabled:  # apply FSDP or HSDP
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)

        dp_mesh = parallel_dims.world_mesh[tuple(dp_mesh_dim_names)]
        
        apply_fsdp_wan(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the Wan model")
        else:
            logger.info("Applied FSDP to the Wan model")

    return model

def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
        CheckpointImpl,
        offload_wrapper as ptd_offload_wrapper,
    )

    # WanVideoModel -> .model (WanVideoForConditionalGeneration) -> .model (WanDitModel) -> .blocks
    # wan_dit = model.model.dit
    wan_dit = model
    
    if hasattr(wan_dit, "blocks"):
        for layer_id, block in wan_dit.blocks.named_children():
            # TODO: Check config for mode (full vs selective vs offload?)
            # Here keeping it simple like Flux: simple wrapper.
            # If offload is needed, use offload_wrapper.
            
            # Use ptd_checkpoint_wrapper
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
            wan_dit.blocks.register_module(layer_id, block)

        logger.info(f"Applied {ac_config.mode} activation checkpointing to the Wan model")

def apply_fsdp_wan(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Model structure:
    # model (WanVideoModel) -> model.model (WanVideoForConditionalGeneration)
    # -> .text_encoder (WanTextEncoder)
    #    -> .model (T5EncoderModel or T5Stack?) -> .encoder -> .block (T5Block)
    # -> .vae (WanVideoVAE)
    #    -> .encoder, .decoder -> blocks
    # -> .model (WanDitModel) -> .blocks (DiTBlock)
    
    # wan_model = model.model # WanVideoForConditionalGeneration
    wan_model = model
    
    # 1. Shard DiT Blocks
    # WanDitModel is wan_model.model
    if hasattr(wan_model, "blocks"):
        logger.info("Sharding Wan DiT blocks")
        assert len(wan_model.blocks) > 0, "No DiT blocks found to shard."
        for block in wan_model.blocks:
            fully_shard(block, **fsdp_config)
    
    # 2. Shard VAE Blocks (if trainable/heavy)
    # NOTE: zirui, Disabled because we use vae.encode() which bypasses FSDP's forward/pre-forward hooks
    # for gathering parameters. This leads to mixed DTensor/Tensor errors.
    # Since VAE is frozen and runs in no_grad, we can just replicate it (default behavior if not sharded).
    # if hasattr(wan_model, "vae"):
    #     # Recursively shard blocks if they exist.
    #     # wan_video_vae.py has: Encoder3d -> downsamples (ResidualBlock/AttentionBlock), middle, head
        
    #     # Helper to shard sequences
    #     def shard_sequence(seq):
    #         for layer in seq:
    #             # Shard if it has parameters
    #             if any(p.requires_grad for p in layer.parameters()) or True: # Shard even frozen for memory?
    #                  pass
    #             fully_shard(layer, **fsdp_config)
        
    #     # Check encoder
    #     if hasattr(wan_model.vae, "encoder"):
    #          # encoder has downsamples (Sequential), middle (Sequential)
    #          if hasattr(wan_model.vae.encoder, "downsamples"):
    #              shard_sequence(wan_model.vae.encoder.downsamples)
    #          if hasattr(wan_model.vae.encoder, "middle"):
    #              shard_sequence(wan_model.vae.encoder.middle)
    #          fully_shard(wan_model.vae.encoder, **fsdp_config)
             
    #     # Check decoder
    #     if hasattr(wan_model.vae, "decoder"):
    #          if hasattr(wan_model.vae.decoder, "upsamples"):
    #              shard_sequence(wan_model.vae.decoder.upsamples)
    #          if hasattr(wan_model.vae.decoder, "middle"):
    #              shard_sequence(wan_model.vae.decoder.middle)
    #          fully_shard(wan_model.vae.decoder, **fsdp_config)
             
    #     fully_shard(wan_model.vae, **fsdp_config)

    # 3. Shard Text Encoder (T5)
    # if hasattr(wan_model, "text_encoder"):
    #     text_enc = wan_model.text_encoder
    #     if hasattr(text_enc, "model") and hasattr(text_enc.model, "encoder") and hasattr(text_enc.model.encoder, "block"):
    #         logger.info("Sharding Wan text encoder T5 blocks")
    #         assert len(text_enc.model.encoder.block) > 0, "No T5 blocks found to shard."
    #         for block in text_enc.model.encoder.block:
    #             fully_shard(block, **fsdp_config)
        
    #     fully_shard(text_enc, **fsdp_config)

    # 4. Shard root
    fully_shard(model, **fsdp_config)
