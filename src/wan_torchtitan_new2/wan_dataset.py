"""
WanNew2 TorchTitan dataloader.

We intentionally keep this close to `src/wan_torchtitan/wan_dataset.py`:
- reuse OmniFlow `WanVideoDataset` + `WanVideoDataProcessor`
- wrap collator to match TorchTitan Trainer signature: (input_dict, labels)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config import JobConfig

from omniflow.data import DatasetConfig, WanVideoDataProcessor, WanVideoDataset


@dataclass(frozen=True)
class _DatasetEntry:
    dataset_path: str
    dataset_format: str
    data_folder: str = ""


MM_DATASETS: dict[str, _DatasetEntry] = {
    "example_video": _DatasetEntry(
        dataset_path="/root/zirui/data/example_video_dataset/metadata.jsonl",
        dataset_format="jsonl",
        data_folder="",
    ),
    "ultravideo": _DatasetEntry(
        # dataset_path="/root/zirui/data/UltraVideo/clips_short_1920_brief_example.jsonl",
        dataset_path="/root/zirui/data/UltraVideo/clips_short_1920_brief_example_1k.jsonl",
        dataset_format="jsonl",
        data_folder="/root/zirui/data/UltraVideo/clips_short_1920/clips_short_1920",
    ),
    "vidgen-1m": _DatasetEntry(
        dataset_path="/root/zirui/data/VIDGEN-1M/meta_data0.json",
        dataset_format="json",
        data_folder="/root/zirui/data/VIDGEN-1M/VidGen_video_0_clean",
    ),
}


class WanCollatorWrapper:
    """
    Wrap OmniFlow raw batch collation into TorchTitan's (input_dict, labels) convention.

    `WanVideoDataset.get_collator()` returns `RawBatchCollator`, which outputs `list[dict]`.
    We must run `WanVideoDataProcessor.prepare_batch()` to produce tensors:
      - video: [B,C,T,H,W]
      - input_ids / attention_mask
    """

    def __init__(self, raw_collator, processor_config: dict):
        self.raw_collator = raw_collator
        self.processor_config = processor_config
        self._processor: WanVideoDataProcessor | None = None

    def _get_processor(self) -> WanVideoDataProcessor:
        if self._processor is None:
            self._processor = WanVideoDataProcessor(self.processor_config)
            self._processor.build()
        return self._processor

    def __call__(self, batch):
        raw_batch = self.raw_collator(batch)  # list[dict]
        processor = self._get_processor()

        # Build model-ready CPU tensors in dataloader workers.
        batch_out = processor.prepare_batch(
            batch=raw_batch, device=torch.device("cpu"), dtype=torch.float32
        )
        video = batch_out.pop("video")  # [B,C,T,H,W]

        input_dict = {"input": video}
        input_dict.update(batch_out)

        # diffusion target is computed in Trainer; keep labels as dummy
        labels = torch.zeros(1)
        return input_dict, labels


def _default_processor_config():
    # Keep the exact defaults used in existing wan_torchtitan to reduce migration friction.
    return {
        "text_tokenizer": "/zirui/models/umt5-xxl",
        "max_text_length": 512,
        "padding_strategy": "max_length",
        "extra_kwargs": {
            "do_resize": True,
            "size": {"height": 480, "width": 832},
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    }


def _infer_format_from_path(path: str) -> str:
    if path.endswith(".jsonl"):
        return "jsonl"
    if path.endswith(".json"):
        return "json"
    return "jsonl"


def build_wan_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,  # Unused (tokenization is done in OmniFlow processor)
    job_config: JobConfig,
    *,
    num_workers: int = 4,
) -> BaseDataLoader:
    dataset_name = (job_config.training.dataset or "").lower()

    # Optional override: allow treating `training.dataset` as a direct path
    dataset_path: Optional[str] = getattr(job_config.training, "dataset_path", None)
    dataset_format: Optional[str] = None
    data_folder: str = ""

    if dataset_name in MM_DATASETS:
        entry = MM_DATASETS[dataset_name]
        dataset_path = entry.dataset_path
        dataset_format = entry.dataset_format
        data_folder = entry.data_folder
    else:
        # Fallback: assume it's a path
        if dataset_path is None:
            dataset_path = job_config.training.dataset
        dataset_format = _infer_format_from_path(dataset_path)

    processor_config = getattr(job_config, "processor", None) or _default_processor_config()
    processor = WanVideoDataProcessor(processor_config)
    processor.build()

    dataset_cfg = DatasetConfig(
        dataset_type="vision",
        dataset_format=dataset_format,
        data_folder=data_folder,
        dataset_path=dataset_path,
        video_sampling_strategy="frame_num",
        frame_num=81,
        shuffle=False,
        video_backend="imageio",
        processor_config=processor_config,
    )

    dataset = WanVideoDataset(processor=processor, config=dataset_cfg)
    dataset.build()

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
    )

    collator = WanCollatorWrapper(dataset.get_collator(), processor_config=processor_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=job_config.training.local_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    return dataloader

