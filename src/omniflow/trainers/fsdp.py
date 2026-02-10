"""
Native torch FSDP trainer

Inherits common logic from BaseNativeTrainer and adds:
  - FSDP (FullyShardedDataParallel) wrapping
  - Gradient sync via model.no_sync()
  - FSDP-specific gradient clipping
  - DiT safetensors checkpoint saving
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from functools import partial
from importlib import import_module

import torch
import torch.distributed as dist
from loguru import logger
from safetensors.torch import save_file as safe_save_file
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from omniflow.registry import register_trainer
from omniflow.distributed import setup_distributed

from .base import BaseNativeTrainer


class FSDPTrainer(BaseNativeTrainer):
    def __init__(
        self,
        model,
        args,
        train_dataset,
        data_collator,
        processing_class,
        rank,
        world_size,
        local_rank,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    # ------------------------------------------------------------------ #
    #                       Parallelism                                    #
    # ------------------------------------------------------------------ #

    def _apply_parallelism(self):
        """Set up FSDP wrapping."""
        # bf16 casting before device placement
        if self.args.get("bf16", False) and hasattr(self.model, "dit") and self.model.dit is not None:
            self.model.dit.to(dtype=torch.bfloat16)
            if self.rank == 0:
                logger.info("FSDP+bf16: casted `dit` to bf16")

        self.model.to(self.device)
        self._wrap_fsdp()

    def _wrap_fsdp(self):
        # Avoid FSDP wrapping for single GPU runs.
        if int(getattr(self, "world_size", 1)) == 1 and self.args.get("fsdp_disable_on_single_gpu", True):
            if self.rank == 0:
                logger.info(
                    "world_size=1: skipping FSDP wrapping for speed (set fsdp_disable_on_single_gpu=false to override)"
                )
            core_model = self.model
            if hasattr(core_model, "freeze_except"):
                core_model.freeze_except()
            return

        fsdp_config = self.args.get("fsdp_config", {}) or {}
        sharding_strategy_str = self.args.get("fsdp_sharding_strategy", "FULL_SHARD")
        sharding_strategy = getattr(ShardingStrategy, sharding_strategy_str, ShardingStrategy.FULL_SHARD)

        mixed_precision_dtype = self._resolve_dtype()
        mp_policy = None
        if mixed_precision_dtype != torch.float32:
            mp_policy = MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=mixed_precision_dtype,
            )

        backward_prefetch = None
        backward_prefetch_str = fsdp_config.get("backward_prefetch")
        if isinstance(backward_prefetch_str, str):
            backward_prefetch_str = backward_prefetch_str.lower()
            if backward_prefetch_str == "backward_pre":
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
            elif backward_prefetch_str == "backward_post":
                backward_prefetch = BackwardPrefetch.BACKWARD_POST

        ignored_modules = []
        ignored_spec = fsdp_config.get("ignored_modules")
        if ignored_spec:
            try:
                if isinstance(ignored_spec, (list, tuple, set)):
                    ignore_names = {str(item).strip() for item in ignored_spec if str(item).strip()}
                    for name, module in self.model.named_modules():
                        if name in ignore_names:
                            ignored_modules.append(module)
                else:
                    regex = re.compile(str(ignored_spec))
                    for name, module in self.model.named_modules():
                        if regex.fullmatch(name):
                            ignored_modules.append(module)
            except Exception as exc:
                logger.warning(f"Invalid ignored_modules spec: {exc}")

        auto_wrap_policy = None
        layer_cls = self.args.get("fsdp_transformer_layer_cls_to_wrap")
        if layer_cls:
            try:
                if isinstance(layer_cls, (list, tuple)):
                    class_paths = [str(item).strip() for item in layer_cls if str(item).strip()]
                else:
                    class_paths = [name.strip() for name in str(layer_cls).split(",") if name.strip()]
                cls_set = set()
                for class_path in class_paths:
                    if "." not in class_path:
                        # Resolve by class name in the model graph.
                        resolved = False
                        for _, module in self.model.named_modules():
                            if module.__class__.__name__ == class_path:
                                cls_set.add(module.__class__)
                                resolved = True
                        if not resolved:
                            logger.warning(f"fsdp_transformer_layer_cls_to_wrap: could not resolve '{class_path}'")
                        continue
                    module_path, cls_name = class_path.rsplit(".", 1)
                    module = import_module(module_path)
                    cls_set.add(getattr(module, cls_name))
                if cls_set:
                    auto_wrap_policy = partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=cls_set,
                    )
            except Exception as exc:
                logger.warning(f"Failed to set auto_wrap_policy: {exc}")

        self.model = FSDP(
            self.model,
            device_id=self.device,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            use_orig_params=fsdp_config.get("use_orig_params", True),
            forward_prefetch=fsdp_config.get("forward_prefetch", True),
            backward_prefetch=backward_prefetch,
            sync_module_states=fsdp_config.get("sync_module_states", False),
            limit_all_gathers=fsdp_config.get("limit_all_gathers", False),
            auto_wrap_policy=auto_wrap_policy,
            ignored_modules=ignored_modules if ignored_modules else None,
            cpu_offload=(CPUOffload(offload_params=True) if fsdp_config.get("cpu_offload", False) else None),
        )
        # Freeze non-trainable modules after wrapping
        core_model = self.model.module if isinstance(self.model, FSDP) else self.model
        if hasattr(core_model, "freeze_except"):
            core_model.freeze_except()

    # ------------------------------------------------------------------ #
    #                       Gradient sync                                  #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _grad_sync_context(self, is_update_step: bool):
        if isinstance(self.model, FSDP) and not is_update_step:
            with self.model.no_sync():
                yield
        else:
            yield

    def _clip_grad_norm(self) -> float:
        if self.max_grad_norm > 0:
            if isinstance(self.model, FSDP):
                norm = self.model.clip_grad_norm_(self.max_grad_norm)
            else:
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            return norm.item() if isinstance(norm, torch.Tensor) else float(norm)
        return 0.0

    # ------------------------------------------------------------------ #
    #                       Checkpointing                                  #
    # ------------------------------------------------------------------ #

    def _save_dit(self, save_path: str) -> None:
        core_model = self.model.module if isinstance(self.model, FSDP) else self.model
        if not hasattr(core_model, "dit"):
            logger.warning("save_model: model has no `dit` attribute; skipping save.")
            return

        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_state = self.model.state_dict()
            dit_state_dict = {k[len("dit.") :]: v for k, v in full_state.items() if k.startswith("dit.")}
        else:
            dit_state_dict = core_model.dit.state_dict()

        if not dit_state_dict:
            logger.warning("save_model: DiT state dict is empty; skipping save.")
            return

        safe_save_file(dit_state_dict, save_path)
        logger.info(f"Saved DiT weights to {save_path}")

        if hasattr(core_model, "config"):
            try:
                core_model.config.save_pretrained(self.output_dir)
            except Exception as exc:
                logger.warning(f"save_model: failed to save config: {exc}")

    def _save_checkpoint(self):
        if self.rank == 0:
            save_path = os.path.join(
                self.output_dir,
                f"dit_checkpoint-{self.global_step}.safetensors",
            )
            self._save_dit(save_path)

    def save_model(self):
        if self.rank == 0:
            save_path = os.path.join(self.output_dir, "dit_model.safetensors")
            self._save_dit(save_path)
        if dist.is_initialized():
            dist.destroy_process_group()


@register_trainer("fsdp")
def build_fsdp_trainer(*, model, dataset, processor, trainer_args: dict):
    rank, world_size, local_rank = setup_distributed()
    if rank == 0:
        logger.info("Building native FSDP trainer")

    return FSDPTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        data_collator=dataset.get_collator(),
        processing_class=processor,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
