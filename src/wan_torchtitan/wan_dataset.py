"""
TODO:
    1. Implement custom processor for different dataset
    2. Add dataset/dataloader for validation
"""

from typing import Callable, Any
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from datasets import Dataset, load_dataset

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config import JobConfig
from omniflow.data import WanVideoDataset, WanVideoDataProcessor, DatasetConfig


def get_dataset(dataset_path, dataset_format, data_folder=""):
    # dataset_path = job_config.training.dataset
    # if not dataset_path:
    #     dataset_path = "/root/zirui/data/example_video_dataset/metadata.jsonl"

    processor_config = {
        "processor_name": "WanVideo/Wan2.1-T2V-3B",
        "processor_type": "wanvideo",
        "extra_kwargs": {
            "do_resize": True,
            "size": {"height": 480, "width": 832},
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    }

    processor = WanVideoDataProcessor(processor_config)
    processor.build()

    print(f"haha {dataset_path=}", flush=True)

    dataset_cfg = DatasetConfig(
        dataset_type="vision",
        dataset_format=dataset_format,
        data_folder=data_folder,
        dataset_path=dataset_path,
        video_sampling_strategy="frame_num",
        frame_num=49,
        shuffle=False,
        video_backend="qwen_vl_utils",
        processor_config=processor_config,
    )
    return dataset_path, processor, dataset_cfg


def get_example_video_dataset():
    return get_dataset("/root/zirui/data/example_video_dataset/metadata.jsonl", "jsonl")


def get_vidgen_1m_dataset():
    return get_dataset(
        "/root/zirui/data/VIDGEN-1M/meta_data0.json",
        "json",
        "/root/zirui/data/VIDGEN-1M/VidGen_video_0_clean",
    )


MM_DATASETS = {
    "example_video": get_example_video_dataset(),
    "vidgen-1m": get_vidgen_1m_dataset(),
}


def get_dataset_by_name(dataset_name):
    if dataset_name not in MM_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(MM_DATASETS.keys())}"
        )
    return MM_DATASETS[dataset_name]


def _validate_mm_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in MM_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(MM_DATASETS.keys())}"
        )

    config = MM_DATASETS[dataset_name]
    path = dataset_path or config.path
    # logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


# Dataloader Helper
class WanCollatorWrapper:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, batch):
        batch_out = self.collator(batch)

        video = batch_out.pop("video")
        input_dict = {"input": video}
        input_dict.update(batch_out)  # Put rest as auxiliary

        # Create dummy labels because diffusion target is computed in forward
        labels = torch.zeros(1)

        return input_dict, labels


def build_wan_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,  # Unused
    job_config: JobConfig,
) -> BaseDataLoader:
    # dataset_path = job_config.training.dataset
    # if not dataset_path:
    #     dataset_path = "/root/zirui/data/example_video_dataset/metadata.jsonl"

    # processor_config = {
    #     "processor_name": "WanVideo/Wan2.1-T2V-3B",
    #     "processor_type": "wanvideo",
    #     "extra_kwargs": {
    #         "do_resize": True,
    #         "size": {"height": 480, "width": 832},
    #         "do_normalize": True,
    #         "image_mean": [0.5, 0.5, 0.5],
    #         "image_std": [0.5, 0.5, 0.5],
    #     }
    # }

    # processor = WanVideoDataProcessor(processor_config)
    # processor.build()

    # dataset_cfg = DatasetConfig(
    #     dataset_type="vision",
    #     dataset_format="jsonl",
    #     data_folder="None",
    #     dataset_path=dataset_path,
    #     video_sampling_strategy="frame_num",
    #     frame_num=49,
    #     shuffle=False,
    #     video_backend="qwen_vl_utils",
    #     processor_config=processor_config
    # )

    # Get dataset by name
    dataset_name = job_config.training.dataset
    dataset_path, processor, dataset_cfg = get_dataset_by_name(dataset_name)

    dataset = WanVideoDataset(
        data_path=dataset_path, processor=processor, config=dataset_cfg
    )
    dataset.build()


    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
    )

    # Wrap collator
    collator = WanCollatorWrapper(dataset.get_collator())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=job_config.training.local_batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
    )

    return dataloader


class WanDataset(IterableDataset, Stateful):
    """Dataset for wan text-to-video model.

    Args:
    dataset_name (str): Name of the dataset.
    dataset_path (str): Path to the dataset.
    model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
    dp_rank (int): Data parallel rank.
    dp_world_size (int): Data parallel world size.
    infinite (bool): Whether to loop over the dataset infinitely.
    """

    def __init__(
        self,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        job_config: JobConfig,
        dataset_name: str,
        dataset_path: str | None = None,
        infinite: bool = False,
    ):
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._t5_tokenizer = t5_tokenizer
        self._t5_empty_token = t5_tokenizer.encode("")
        self._clip_tokenizer = clip_tokenizer
        self._clip_empty_token = clip_tokenizer.encode("")
        self._data_processor = data_processor
        self.job_config = job_config

        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_samples: list[dict[str, Any]] = []

    def __iter__(self):
        dataset_iterator = self._get_data_iter()
