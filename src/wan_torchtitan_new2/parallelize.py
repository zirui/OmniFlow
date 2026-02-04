# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_wan_new2(model: nn.Module, parallel_dims: ParallelDims, job_config: JobConfig):
    """
    Apply TorchTitan parallelism to wan_new2 DiT model.

    Notes:
    - We only shard the DiT backbone (model.dit).
    - VAE/T5 live outside the model in the Trainer, so no need to shard encoders here.
    """

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if parallel_dims.dp_shard_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)

        dp_mesh = parallel_dims.world_mesh[tuple(dp_mesh_dim_names)]
        apply_fsdp_wan_new2(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to wan_new2 DiT")
        else:
            logger.info("Applied FSDP to wan_new2 DiT")

    return model


def apply_ac(model: nn.Module, ac_config):
    """
    Apply activation checkpointing wrapper on DiT blocks.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )

    if not hasattr(model, "dit") or not hasattr(model.dit, "blocks"):
        logger.warning("wan_new2.apply_ac: model.dit.blocks not found, skip.")
        return

    for layer_id, block in model.dit.blocks.named_children():
        wrapped = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.dit.blocks.register_module(layer_id, wrapped)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to wan_new2 DiT blocks")


def apply_fsdp_wan_new2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_policy: str = "default",
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # torch.distributed.fsdp.fully_shard supports reshard_after_forward kwarg
    def _reshard_kw(is_last: bool = False) -> dict:
        if reshard_policy == "never":
            return {"reshard_after_forward": False}
        if reshard_policy == "always":
            return {"reshard_after_forward": True}
        # default
        return {"reshard_after_forward": False} if is_last else {}

    dit = getattr(model, "dit", None)
    if dit is None:
        raise AttributeError("wan_new2 model must have `.dit` attribute")

    # Shard embeddings / small modules first (optional)
    for name in ("patch_embedding", "text_embedding", "time_embedding", "time_projection"):
        m = getattr(dit, name, None)
        if m is not None:
            fully_shard(m, **fsdp_config)

    # Shard transformer blocks
    if hasattr(dit, "blocks"):
        for block in dit.blocks:
            fully_shard(block, **fsdp_config)

    # Shard head last, disable reshard-after-forward if requested
    head = getattr(dit, "head", None)
    if head is not None:
        fully_shard(head, **fsdp_config, **_reshard_kw(is_last=True))

    # Shard root of DiT wrapper and the outer model
    fully_shard(dit, **fsdp_config)
    fully_shard(model, **fsdp_config)

