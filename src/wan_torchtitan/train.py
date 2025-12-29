# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn

from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer
from torchtitan.distributed import ParallelDims

from torchtitan.experiments.wan.wan_args import WanModelArgs
from torchtitan.experiments.wan.wan_model import WanVideoModel
from torchtitan.experiments.wan.parallelize import parallelize_wan
from torchtitan.experiments.wan.loss import build_wan_loss
from torchtitan.experiments.wan.wan_dataset import build_wan_dataloader

class WanTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)
        # TOOD: zirui, Re-initialize deterministic mode if needed (Flux does it for specific reasons, we might too)

        logger.info(f"Building Wan Model using local experiment definition")
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )
        
        # Parse Args
        # model_args = WanModelArgs()
        model_args = self.train_spec.model_args[job_config.model.flavor]
        print(f"haha, {model_args=}", flush=True)
        # model_args.update_from_config(job_config)
        # self.model_args = model_args
        
        # init_device = "cpu" # or "meta" if we supported it fully. 
        
        # clean up existing model from Trainer init (if any)
        if hasattr(self, 'model_parts'):
            del self.model_parts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Building custom Wan model...")
        
        with utils.set_default_dtype(self._dtype):
             model = WanVideoModel(model_args)
        
        # Parallelize
        model = parallelize_wan(model, self.parallel_dims, job_config)
        
        # Move to device and init weights
        model.to(device=self.device)
        
        # We can access model.module (if FSDP)
        if isinstance(model, torch.distributed.fsdp.FullyShardedDataParallel):
            inner_model = model.module
        else:
            inner_model = model
            
        if hasattr(inner_model, "init_weights"):
            inner_model.init_weights()

        model.train()
        self.model_parts = [model]
        
        # Update Optimizers/Schedulers for new model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config.optimizer, self.parallel_dims, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )
        
        # Override Dataloader
        self.dataloader = build_wan_dataloader(
            dp_world_size=self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard, # approx dp degree
            dp_rank =torch.distributed.get_rank(),
            tokenizer=None,
            job_config=job_config
        )
        
        # Override Loss
        self.loss_fn = build_wan_loss(job_config, self.parallel_dims, self.ft_manager)
        
        # Re-setup checkpointer to track new model/opt
        from torchtitan.components.checkpoint import CheckpointManager
        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=None, # Wan custom adapter if needed
            base_folder=job_config.job.dump_folder,
            ft_manager=self.ft_manager,
        )

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model = self.model_parts[0]
        
        inputs = input_dict.pop("input")
        extra_kwargs = input_dict # Remaining keys
        
        with self.maybe_enable_amp:
            pred = model(inputs, **extra_kwargs)
            
            # Loss computation, TODO: zirui, labels arg is dummy from our collator.
            loss = self.loss_fn(pred, labels)
            
        del pred
        loss.backward()
        return loss


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    
    # Force disable validation if not ready
    # config.validation.enable = False
    
    trainer: Optional[WanTrainer] = None

    try:
        trainer = WanTrainer(config)
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed.")
