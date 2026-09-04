"""Register Wan dataset builder."""

from loguru import logger

from omniflow.data.config import DatasetConfig
from omniflow.data.dataset import WanVideoDataset
from omniflow.data.processor import WanVideoDataProcessor
from omniflow.registry import register_dataset


@register_dataset("wan")
def build_wan_dataset(dataset_config: dict):
    processor_config = dataset_config["processor_config"]
    processor = WanVideoDataProcessor(processor_config)
    processor.build()
    logger.info("Built data processor")

    dataset = WanVideoDataset(
        processor=processor,
        config=DatasetConfig(**dataset_config),
    )
    logger.info(f"Built dataset with {len(dataset)} samples")
    return dataset, processor
