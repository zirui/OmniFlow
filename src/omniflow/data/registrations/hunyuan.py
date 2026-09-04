"""Register HunyuanVideo dataset builder."""

from loguru import logger

from omniflow.data.config import DatasetConfig
from omniflow.data.dataset import WanVideoDataset
from omniflow.data.hunyuan_processor import HunyuanVideoDataProcessor
from omniflow.registry import register_dataset


@register_dataset("hunyuan")
def build_hunyuan_dataset(dataset_config: dict):
    # Keep parity with other builders: require processor_config to exist in YAML.
    processor_config = dataset_config["processor_config"]
    processor = HunyuanVideoDataProcessor(processor_config)
    processor.build()
    logger.info("Built Hunyuan data processor")

    # v0: reuse WanVideoDataset raw I/O and sampling
    dataset = WanVideoDataset(
        processor=processor,
        config=DatasetConfig(**dataset_config),
    )

    logger.info(f"Built hunyuan dataset with {len(dataset)} samples")
    return dataset, processor
