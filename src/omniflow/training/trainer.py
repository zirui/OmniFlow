"""Custom HF trainer for diffusion training.
"""


from typing import Any, Optional, Union
import os

import torch
import torch.nn as nn
from loguru import logger
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from schedulers.flow_match import FlowMatchScheduler
from utils.train_utils import get_memory, set_seed


class WanVideoCallback(TrainerCallback):
    """Callback to freeze non-trainable modules at training start."""
    
    def on_train_begin(self, args, state, control, model=None, logs=None, **kwargs):
        model.freeze_except()
        logger.info(f"Trainable_modules: {model.trainable_modules}. Freezing other modules.")
        
        # Reset memory stats to track training peak instead of initialization peak
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            allocated, reserved, max_alloc = get_memory()
            logger.info(
                f"step={state.global_step}, "
                f"loss={logs['loss']:.4f}, "
                f"allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max_alloc={max_alloc:.2f}GB"
            )
            

class WanVideoTrainer(HFTrainer):
    """
    Custom trainer for WanVideo diffusion training.
    
    Implements flow-matching diffusion training with custom loss computation.
    """
    
    def __init__(self, *args, **kwargs):
        # Add WanVideo callback
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(WanVideoCallback())
        kwargs["callbacks"] = callbacks
        
        super().__init__(*args, **kwargs)
        
        # Initialize flow-matching scheduler
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        logger.info(f"Setting timesteps for diffusion training: {len(self.scheduler.timesteps)} steps")
        
        # Global seeding if FIXED_SEED is set
        if os.environ.get("FIXED_SEED"):
            try:
                # TODO: zirui, fixed global seed for debugging
                seed = int(os.environ["FIXED_SEED"])
                set_seed(seed)
                logger.info(f"Global seed set to {seed}")
            except ValueError:
                raise ValueError("FIXED_SEED must be an integer")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute diffusion training loss.
        
        Args:
            model: WanVideo model
            inputs: Batch inputs with video, input_ids, attention_mask
            num_items_in_batch: Number of items in batch (unused)
            
        Returns:
            Loss tensor
        """
        pixel_values = inputs.get("video")
        # print(f"[DEBUG]{pixel_values.shape=}")
        # "video": pixel_values.squeeze(0),  # T, H, W, C
        if pixel_values.ndim == 5:
            #  [B, C, T, H, W]
            num_frames, height, width = pixel_values.shape[2:5]
        else:
            # [T, H, W, C]
            num_frames, height, width = pixel_values.shape[:3]
        
        # Prepare inputs dict for model
        inputs_dict = {
            "video": pixel_values,
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            # "input_image": pixel_values[:, 0] if pixel_values.ndim == 5 else pixel_values[0],
            "input_image": pixel_values.select(2, 0) if pixel_values.ndim == 5 else pixel_values[0],
            "cfg_scale": inputs.get("cfg_scale", 1),
            "cfg_merge": inputs.get("cfg_merge", False),
            "seed": inputs.get("seed", None), 
            "reference_image": inputs.get("reference_image", None),
            "tiled": inputs.get("tiled", False),
            "tile_size": inputs.get("tile_size", None),
            "tile_stride": inputs.get("tile_stride", None),
            "end_image": inputs.get("end_image", None),
        }
        
        # Override seed if FIXED_SEED is set
        if os.environ.get("FIXED_SEED"):
            try:
                fixed_seed = int(os.environ["FIXED_SEED"])
                # logger.info(f"Using FIXED_SEED: {fixed_seed}") # Commented out to avoid spamming logs
            except ValueError:
                raise ValueError(f"Invalid FIXED_SEED value: {os.environ['FIXED_SEED']}")
            inputs_dict["seed"] = fixed_seed
        
        # Sample random timestep
        if os.environ.get("FIXED_TIMESTEP"):
            try:
                fixed_step = int(os.environ["FIXED_TIMESTEP"])
                max_step = self.scheduler.num_train_timesteps - 1
                if fixed_step < 0 or fixed_step > max_step:
                     logger.warning(f"FIXED_TIMESTEP {fixed_step} out of range [0, {max_step}]. Clamping.")
                     fixed_step = max(0, min(fixed_step, max_step))
                
                timestep_id = torch.tensor([fixed_step], device=self.scheduler.timesteps.device)
                logger.info(f"Using FIXED_TIMESTEP: {fixed_step}")
            except ValueError:
                raise ValueError(f"Invalid FIXED_TIMESTEP value: {os.environ['FIXED_TIMESTEP']}")
        else:
            max_timestep_boundary = int(1 * self.scheduler.num_train_timesteps)
            min_timestep_boundary = int(0 * self.scheduler.num_train_timesteps)
            timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
            
        timestep = self.scheduler.timesteps[timestep_id]

        # Preprocess inputs (encode video, text, etc.)
        if isinstance(model, FSDP):
            with FSDP.summon_full_params(model, writeback=False, rank0_only=False):
                pre_precessed_inputs = model.forward_preprocess(self.scheduler, inputs_dict)
                # Get model dtype for timestep casting
                model_dtype = model.dtype
        else:
            # TODO: zirui, fix ddp bug
            model = model.module if hasattr(model, "module") else model
            pre_precessed_inputs = model.forward_preprocess(self.scheduler, inputs_dict)
            model_dtype = next(model.parameters()).dtype
            
        # Cast timestep to model dtype (MATCHES DiffSynth behavior: 833.33 -> 832.0 if bf16)
        timestep = timestep.to(dtype=model_dtype)

        # Compute training target
        training_target = self.scheduler.training_target(
            pre_precessed_inputs["input_latents"],
            pre_precessed_inputs["noise"],
            timestep,
        )
        
        # Add noise to latents
        pre_precessed_inputs["latents"] = self.scheduler.add_noise(
            pre_precessed_inputs["input_latents"],
            pre_precessed_inputs["noise"],
            timestep,
        )
        
        # Forward pass
        output = model(
            latents=pre_precessed_inputs.get("latents", None),
            context=pre_precessed_inputs.get("context", None),
            timestep=timestep,
            y=pre_precessed_inputs.get("y", None),
            reference_latents=pre_precessed_inputs.get("reference_latents", None),
            clip_feature=pre_precessed_inputs.get("clip_feature", None),
        )
        
        # Compute MSE loss
        noise_pred = output.noise_pred
        
        # TODO: zirui, for debugging 
        if os.getenv("DEBUG") == "1" and self.state.global_step % 1 == 0:
            weight = self.scheduler.training_weight(timestep)
            logger.info(f"DEBUG: Step={self.state.global_step} Timestep={timestep.item():.4f} Weight={weight.item():.4f}")
            logger.info(f"DEBUG: Video Min={pixel_values.min().item():.4f} Max={pixel_values.max().item():.4f}")
            logger.info(f"DEBUG: Pred Mean={noise_pred.mean().item():.4f} Std={noise_pred.std().item():.4f}")
            logger.info(f"DEBUG: Target Mean={training_target.mean().item():.4f} Std={training_target.std().item():.4f}")
            raw_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="mean")
            logger.info(f"DEBUG: Raw MSE={raw_loss.item():.6f}")
            
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="mean")
        loss = loss * self.scheduler.training_weight(timestep)
        return loss
