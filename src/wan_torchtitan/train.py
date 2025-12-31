# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
torch.manual_seed(0)
import torch.nn as nn

from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer
from torchtitan.distributed import ParallelDims

from torchtitan.experiments.wan.wan_args import WanModelArgs
# from torchtitan.experiments.wan.wan_model import WanVideoModel
from torchtitan.experiments.wan.parallelize import parallelize_wan
from torchtitan.experiments.wan.loss import build_wan_loss
from torchtitan.experiments.wan.wan_dataset import build_wan_dataloader

from .model.modeling_wanvideo import WanVideoForConditionalGeneration
from .model.wan_video_scheduler import FlowMatchScheduler
from .model import WanVideoConfig
from .debug_utils import print_tensor
class WanTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)
        model = self.model_parts[0]

        logger.info(f"Building Wan Model using local experiment definition")
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )
        config_dict = {
            k: v
            for k, v in vars(self.model_args).items()
            if k in WanVideoConfig.__annotations__ or k in WanVideoConfig().__dict__
        }
        wan_config = WanVideoConfig(**config_dict)

        #TODO (limou)
        # check determinism
        # ...

        self.data_processor = WanVideoForConditionalGeneration(wan_config)

        # TODO (limou)
        # apply FSDP for vae && text_encoder
        # ...

        self.data_processor.load_model(self.model_args.vae_checkpoint_path, self.model_args.t5_checkpoint_path)
        self.data_processor = self.data_processor.to(self.device)

        self.scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        pass # End of __init__

    def forward_backward_step(
        self, inputs_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        inputs_dict["video"] = inputs_dict.pop("input")
        video = inputs_dict["video"]

        processor_device = next(self.data_processor.parameters()).device
        model_device = next(self.model_parts[0].parameters()).device
        logger.info("video.device={},processor_device={}, model_device={}".format(
            video.device, processor_device, model_device))

        # Add missing keys with defaults if not present
        # [B, C, F, H, W]
        defaults = {
            "input_ids": None,
            "attention_mask": None,
            "cfg_scale": 1,
            "cfg_merge": False,
            "vace_scale": 1,
            "seed": None,
            "vace_reference_image": None,
            "reference_image": None,
            "tiled": False,
            "tile_size": None,
            "tile_stride": None,
            "end_image": None,
            "camera_control_direction": None,
            "camera_control_speed": None,
            "camera_control_origin": None,
            "control_video": None,
            "motion_bucket_id": None,
            "vace_video": None,
            "vace_video_mask": None,
            # "input_image": video[:, 0] if video.ndim == 5 else video[0],
            "input_image": video.select(2, 0) if video.ndim == 5 else video[0],
        }
        for k, v in defaults.items():
            inputs_dict.setdefault(k, v)

        # Recover height/width/num_frames
        if "height" not in inputs_dict:
            if video.ndim == 5:
                # [B, C, F, H, W]
                inputs_dict["num_frames"], inputs_dict["height"], inputs_dict["width"] = (
                    video.shape[2:5]
                )
            else:
                inputs_dict["num_frames"], inputs_dict["height"], inputs_dict["width"] = (
                    video.shape[:3]
                )

        # Sample random timestep
        max_timestep_boundary = int(1 * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(0 * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id]

        with torch.no_grad():
            pre_processed_inputs = self.data_processor.forward_preprocess(
                self.scheduler, inputs_dict
            )

        # Compute training target
        training_target = self.scheduler.training_target(
            pre_processed_inputs["input_latents"],
            pre_processed_inputs["noise"],
            timestep,
        )
        # Add noise
        pre_processed_inputs["latents"] = self.scheduler.add_noise(
            pre_processed_inputs["input_latents"],
            pre_processed_inputs["noise"],
            timestep,
        )

        with self.maybe_enable_amp:
            pred = self.model_parts[0](
                x=pre_processed_inputs.get("latents", None),
                context=pre_processed_inputs.get("context", None),
                timestep=timestep,
                # TODO (limou)
                # attention_mask ?
            )
            
            # Loss computation, TODO: zirui, labels arg is dummy from our collator.
            loss_input = (pred, training_target, timestep)
            loss = self.loss_fn(loss_input, labels)
            
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
