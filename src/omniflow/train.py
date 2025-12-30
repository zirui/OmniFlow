#!/usr/bin/env python3
"""training entry point"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
import torch
from transformers import TrainingArguments

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import WanVideoForConditionalGeneration, WanVideoConfig
from data import WanVideoDataset, WanVideoDataProcessor, DatasetConfig  
from training import WanVideoTrainer
from utils.train_utils import count_parameters

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(model_config: dict):
    """Build WanVideo model from config."""
    if 'load_from_pretrained_path' in model_config:
        # Load from pretrained
        pretrained_path = model_config['load_from_pretrained_path']
        
        if 'config' in model_config:
             # Load using custom load_dit with config overrides
            logger.info(f"Loading DiT model from {pretrained_path} with config overrides")
            config_dict = model_config['config']
            # Pass config params as kwargs
            model = WanVideoForConditionalGeneration.load_dit(pretrained_path, **config_dict)
        else:
            try:
                logger.info(f"Attempting to load as DiT checkpoint from {pretrained_path}")
                model = WanVideoForConditionalGeneration.load_dit(pretrained_path)
            except Exception as e:
                logger.warning(f"load_dit failed: {e}. Falling back to standard from_pretrained.")
                model = WanVideoForConditionalGeneration.from_pretrained(pretrained_path)

        # Update trainable modules if specified
        if 'trainable_modules' in model_config:
            model.config.trainable_modules = model_config['trainable_modules']
            model.trainable_modules = model_config['trainable_modules']            

            
    elif 'config' in model_config:
        # Build from scratch
        logger.info("Building model from scratch")
        logger.debug(f"{model_config=}")
        config_dict = model_config['config']
        model_cfg = WanVideoConfig(**config_dict)
        model = WanVideoForConditionalGeneration(model_cfg)
    else:
        raise ValueError("model_config must contain either 'load_from_pretrained_path' or 'config'")

    logger.info(f"Model Structure: {model}")

    total_params, trainable_params = count_parameters(model)
    logger.info(f"parameters: total_params={total_params/1e9:.2f}B, trainable_params={trainable_params/1e9:.2f}B")
    
    # Freeze modules
    model.freeze_except()
    total_params, trainable_params = count_parameters(model)
    logger.info(f"parameters after freezing: total_params={total_params/1e9:.2f}B, trainable_params={trainable_params/1e9:.2f}B")
    
    logger.info(f"parameters after freezing: total_params={total_params/1e9:.2f}B, trainable_params={trainable_params/1e9:.2f}B")
        
    # Load VAE and Text Encoder Weights
    load_encoder_weights(model, model_config)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"parameters after loading encoder weights: total_params={total_params/1e9:.2f}B, trainable_params={trainable_params/1e9:.2f}B")

    # Cast model to specified dtype
    if 'model_dtype' in model_config:
        dtype_str = model_config['model_dtype']
        logger.info(f"Casting model to {dtype_str}")
        if dtype_str == 'bfloat16':
            model.to(torch.bfloat16)
        elif dtype_str == 'float16':
            model.to(torch.float16)
        elif dtype_str == 'float32':
            model.to(torch.float32)

    # logger.info("Applying CPU Offloading: Casting encoders to bfloat16 and concealing from FSDP...")
     # # --- CPU Offloading & FSDP Hiding ---
    
    # # 1. Text Encoder
    # if hasattr(model, 'text_encoder') and model.text_encoder is not None:
    #     model.text_encoder.to(torch.bfloat16).to("cpu")
    #     model.text_encoder_hidden = [model.text_encoder]
    #     del model.text_encoder
    #     logger.info("Offloaded text_encoder to CPU (bfloat16) and hidden from FSDP")

    # # 2. VAE
    # if hasattr(model, 'vae') and model.vae is not None:
    #     model.vae.to(torch.bfloat16).to("cpu")
    #     model.vae_hidden = [model.vae]
    #     del model.vae
    #     logger.info("Offloaded vae to CPU (bfloat16) and hidden from FSDP")
        
    # # 3. Image Encoder (if present)
    # if hasattr(model, 'image_encoder') and model.image_encoder is not None:
    #     model.image_encoder.to(torch.bfloat16).to("cpu")
    #     model.image_encoder_hidden = [model.image_encoder]
    #     del model.image_encoder
    #     logger.info("Offloaded image_encoder to CPU (bfloat16) and hidden from FSDP")
        
    return model


def build_dataset(dataset_config: dict):
    """Build dataset and processor from config."""
    # Build processor
    processor_config = dataset_config['processor_config']
    processor = WanVideoDataProcessor(processor_config)
    processor.build()
    logger.info("Built data processor")
    
    logger.debug(f"{dataset_config=}")
    # Build dataset
    dataset = WanVideoDataset(
        processor=processor,
        config = DatasetConfig(**dataset_config)
    )
    dataset.build()
    logger.info(f"Built dataset with {len(dataset)} samples")
    
    return dataset, processor


def load_encoder_weights(model, config):
    if 'encoder' not in config:
        return
    
    encoder_cfg = config['encoder']
    
    # Load T5 Encoder
    if 't5_encoder' in encoder_cfg:
        t5_path = encoder_cfg['t5_encoder']
        logger.info(f"Loading T5 Encoder from {t5_path}")
        try:
            state_dict = torch.load(t5_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Helper to remove prefix if needed (e.g. if saved from DDP)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

            # Check if model.text_encoder handles the loading or we load directly
            keys_result = model.text_encoder.load_state_dict(state_dict, strict=False)
            logger.info(f"T5 Encoder loaded. Missing keys: {len(keys_result.missing_keys)}, Unexpected keys: {len(keys_result.unexpected_keys)}")
        except Exception as e:
            logger.error(f"Failed to load T5 Encoder: {e}")
            
    # Load Autoencoder (VAE)
    if 'autoencoder' in encoder_cfg:
        vae_path = encoder_cfg['autoencoder']
        logger.info(f"Loading VAE from {vae_path}")
        try:
            state_dict = torch.load(vae_path, map_location='cpu')
            
            # Using VAE's converter logic to likely handle prefix mismatch
            # The VAE class in model_vae.py has a converter that prepends 'model.'
            from models.wan.wan_video_vae import WanVideoVAE
            
            converter = WanVideoVAE.state_dict_converter()
            state_dict = converter.from_civitai(state_dict)
            
            keys_result = model.vae.load_state_dict(state_dict, strict=False)
            logger.info(f"VAE loaded. Missing keys: {len(keys_result.missing_keys)}, Unexpected keys: {len(keys_result.unexpected_keys)}")
            
        except Exception as e:
             logger.error(f"Failed to load VAE: {e}")

def main():
    parser = argparse.ArgumentParser(description="WanVideo Standalone Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Build model
    model = build_model(config['model_config'])
    logger.info(f"Model built successfully")
    
    # Build dataset
    train_dataset, processor = build_dataset(config['dataset_config'])
    
    # Build training arguments
    trainer_args_dict = config['trainer_args']
    training_args = TrainingArguments(**trainer_args_dict)
    logger.info(f"Training arguments: {training_args}")
    
    # Build trainer
    trainer = WanVideoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_dataset.get_collator(),
        processing_class=processor,
    )
    logger.info("Trainer built successfully")
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to: {training_args.output_dir}")
    trainer.save_model()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
