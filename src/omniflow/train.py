#!/usr/bin/env python3
"""training entry point"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from transformers import TrainingArguments

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import WanVideoForConditionalGeneration, WanVideoConfig
from data import WanVideoDataset, WanVideoDataProcessor, DatasetConfig  
from training import WanVideoTrainer


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
        logger.info(f"Loading model from pretrained: {pretrained_path}")
        model = WanVideoForConditionalGeneration.from_pretrained(pretrained_path)
        
        # Update trainable modules if specified
        if 'trainable_modules' in model_config:
            model.config.trainable_modules = model_config['trainable_modules']
            model.trainable_modules = model_config['trainable_modules']
            
    elif 'config' in model_config:
        # Build from scratch
        logger.info("Building model from scratch")
        config_dict = model_config['config']
        model_cfg = WanVideoConfig(**config_dict)
        model = WanVideoForConditionalGeneration(model_cfg)
    else:
        raise ValueError("model_config must contain either 'load_from_pretrained_path' or 'config'")
    
    return model


def build_dataset(dataset_config: dict):
    """Build dataset and processor from config."""
    # Build processor
    processor_config = dataset_config['processor_config']
    processor = WanVideoDataProcessor(processor_config)
    processor.build()
    logger.info("Built data processor")
    
    # Build dataset
    dataset = WanVideoDataset(
        data_path=dataset_config['dataset_path'],
        processor=processor,
        config = DatasetConfig(**dataset_config)
    )
    dataset.build()
    logger.info(f"Built dataset with {len(dataset)} samples")
    
    return dataset, processor


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
