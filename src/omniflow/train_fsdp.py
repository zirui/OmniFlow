#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import time
import math
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import WanVideoForConditionalGeneration, WanVideoConfig
from data import WanVideoDataset, WanVideoDataProcessor, DatasetConfig
from training.scheduler import FlowMatchScheduler
from loguru import logger

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model(model_config: dict):
    if 'load_from_pretrained_path' in model_config:
        pretrained_path = model_config['load_from_pretrained_path']
        logger.info(f"Loading model from pretrained: {pretrained_path}")
        model = WanVideoForConditionalGeneration.from_pretrained(pretrained_path)
        if 'trainable_modules' in model_config:
            model.config.trainable_modules = model_config['trainable_modules']
            model.trainable_modules = model_config['trainable_modules']
    elif 'config' in model_config:
        logger.info("Building model from scratch")
        config_dict = model_config['config']
        model_cfg = WanVideoConfig(**config_dict)
        model = WanVideoForConditionalGeneration(model_cfg)
    else:
        raise ValueError("model_config must contain either 'load_from_pretrained_path' or 'config'")
    return model

def build_dataset(dataset_config: dict):
    processor_config = dataset_config['processor_config']
    processor = WanVideoDataProcessor(processor_config)
    processor.build()
    logger.info("Built data processor")
    
    dataset = WanVideoDataset(
        data_path=dataset_config['dataset_path'],
        processor=processor,
        config=DatasetConfig(**dataset_config)
    )
    dataset.build()
    logger.info(f"Built dataset with {len(dataset)} samples")
    return dataset, processor

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(0)
    return rank, world_size, local_rank

def compute_loss(model, scheduler, batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
            
    pixel_values = batch.get("video")
    if pixel_values is None:
         raise ValueError("Batch must contain 'video' key")

    num_frames, height, width = pixel_values.shape[:3]
    
    inputs_dict = {
        "video": pixel_values,
        "input_ids": batch.get("input_ids"),
        "attention_mask": batch.get("attention_mask"),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "input_image": pixel_values[0],
        "cfg_scale": batch.get("cfg_scale", 1),
        "cfg_merge": batch.get("cfg_merge", False),
        "vace_scale": batch.get("vace_scale", 1),
        "seed": batch.get("seed", None),
        "vace_reference_image": batch.get("vace_reference_image", None),
        "reference_image": batch.get("reference_image", None),
        "tiled": batch.get("tiled", False),
        "tile_size": batch.get("tile_size", None),
        "tile_stride": batch.get("tile_stride", None),
        "end_image": batch.get("end_image", None),
        "camera_control_direction": batch.get("camera_control_direction", None),
        "camera_control_speed": batch.get("camera_control_speed", None),
        "camera_control_origin": batch.get("camera_control_origin", None),
        "control_video": batch.get("control_video", None),
        "motion_bucket_id": batch.get("motion_bucket_id", None),
        "vace_video": batch.get("vace_video", None),
        "vace_video_mask": batch.get("vace_video_mask", None),
    }

    max_timestep_boundary = int(1 * scheduler.num_train_timesteps)
    min_timestep_boundary = int(0 * scheduler.num_train_timesteps)
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,), device=device)
    timestep = scheduler.timesteps[timestep_id.cpu()]
    timestep = timestep.to(device)

    with FSDP.summon_full_params(model, writeback=False, rank0_only=False):
         pre_processed_inputs = model.forward_preprocess(scheduler, inputs_dict)
         
    training_target = scheduler.training_target(
        pre_processed_inputs["input_latents"],
        pre_processed_inputs["noise"],
        timestep,
    )
    
    pre_processed_inputs["latents"] = scheduler.add_noise(
        pre_processed_inputs["input_latents"],
        pre_processed_inputs["noise"],
        timestep,
    )
    
    output = model(
        latents=pre_processed_inputs.get("latents", None),
        context=pre_processed_inputs.get("context", None),
        timestep=timestep,
        y=pre_processed_inputs.get("y", None),
        reference_latents=pre_processed_inputs.get("reference_latents", None),
        clip_feature=pre_processed_inputs.get("clip_feature", None),
        vace_context=pre_processed_inputs.get("vace_context", None),
        vace_scale=pre_processed_inputs.get("vace_scale", 1.0),
        motion_bucket_id=pre_processed_inputs.get("motion_bucket_id", None),
        control_camera_latents_input=pre_processed_inputs.get("control_camera_latents_input", None),
    )
    
    noise_pred = output.noise_pred
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="mean")
    loss = loss * scheduler.training_weight(timestep)
    
    return loss

def main():
    parser = argparse.ArgumentParser(description="WanVideo Native FSDP Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Extract training args
    trainer_args = config.get('trainer_args', {})
    
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        logger.info(f"Starting training with config: {args.config}")
        os.makedirs(trainer_args.get('output_dir', './output'), exist_ok=True)

    # Build Model
    model = build_model(config['model_config'])
    model.freeze_except() # Relies on model config 'trainable_modules' which is set in build_model
    
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    
    # FSDP Configuration
    mixed_precision_dtype = torch.float32
    if trainer_args.get('bf16', False):
        mixed_precision_dtype = torch.bfloat16
    elif trainer_args.get('fp16', False):
        mixed_precision_dtype = torch.float16
        
    mp_policy = MixedPrecision(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    
    sharding_strategy_str = trainer_args.get('fsdp_sharding_strategy', 'FULL_SHARD')
    sharding_strategy = getattr(ShardingStrategy, sharding_strategy_str)
    
    # Wrap model
    model = FSDP(
        model,
        device_id=device,
        mixed_precision=mp_policy if mixed_precision_dtype != torch.float32 else None,
        sharding_strategy=sharding_strategy,
        use_orig_params=True, 
        forward_prefetch=True,
    )
    
    # Build Dataset
    dataset, processor = build_dataset(config['dataset_config'])
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=config['dataset_config'].get('shuffle', False),
    )
    
    batch_size = trainer_args.get('per_device_train_batch_size', 1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=trainer_args.get('dataloader_num_workers', 4),
        collate_fn=dataset.get_collator(),
        pin_memory=True
    )
    
    # Optimizer
    learning_rate = float(trainer_args.get('learning_rate', 5e-5))
    weight_decay = float(trainer_args.get('weight_decay', 0.01))
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True) # TODO: Make timesteps configurable if needed
    
    # Training Loop
    model.train()
    step = 0
    max_steps = trainer_args.get('max_steps', 10000)
    save_steps = trainer_args.get('save_steps', 100)
    logging_steps = trainer_args.get('logging_steps', 1)
    output_dir = trainer_args.get('output_dir', './output')
    
    if rank == 0:
        logger.info("Starting training loop...")
        
    num_train_epochs = trainer_args.get('num_train_epochs', 100)
    
    for epoch in range(num_train_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            
            optimizer.zero_grad()
            
            loss = compute_loss(model, scheduler, batch, device)
            
            loss.backward()
            
            optimizer.step()
            
            if step % logging_steps == 0 and rank == 0:
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
            if step % save_steps == 0 and step > 0:
                if rank == 0:
                    logger.info(f"Saving checkpoint at step {step}")
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                        cpu_state = model.state_dict()
                        if rank == 0:
                            save_path = os.path.join(output_dir, f"checkpoint-{step}.pt")
                            torch.save(cpu_state, save_path)
                            
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break
            
    # Final save
    if rank == 0:
        logger.info("Saving final model...")
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            if rank == 0:
                save_path = os.path.join(output_dir, "final_model.pt")
                torch.save(cpu_state, save_path)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
