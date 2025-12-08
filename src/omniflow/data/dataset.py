"""
Dataset class for loading video data from JSONL/CSV files.
"""

from abc import abstractmethod
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from io import BytesIO
import os
from decord import VideoReader, cpu
from loguru import logger

import torch
from torch.utils.data import Dataset
from omniflow.utils.data_utils import smart_nframes
from omniflow.utils import fetch_video

from .collator import VisionCollator

# class WanVideoDataCollator(VisionCollator):
#     def __init__(self, processor):
#         super().__init__(processor)


class BaseDataset(Dataset):
    # def __init__(self, config: DatasetConfig) -> None:
    def __init__(self, config, **kwargs) -> None:
        """
        Initialize the base dataset with configuration.
        Args:
            config: Dataset configuration object containing all necessary parameters
        """
        super().__init__()
        self.config = config
        # self.processor_config = config.processor_config
        # if isinstance(self.processor_config, dict):
        #     self.processor_config = ProcessorConfig(**self.processor_config)
        self.samples = []
        self.skip = set([19, 20])
        self.valid_indices = [i for i in range(len(self.samples)) if i not in self.skip]

    def __len__(self):
        return len(self.valid_indices)

    def build(self):
        """
        Build the dataset by loading data and building the processor.
        This method should be called after initialization to prepare the dataset.
        """
        # self._build_from_config()
        # self.processor = self._build_processor()
        self.processor.build()

    @abstractmethod
    def _build_from_config(self):
        """
        Load and prepare data from the configuration.

        This method should implement the logic to load data from various sources
        (JSON, JSONL, Arrow, Parquet, HF Dataset, YAML) based on the dataset format
        specified in the configuration.
        """
        pass


class WanVideoDataset(BaseDataset):
    """Dataset for WanVideo training from JSONL/CSV."""
    
    def __init__(
        self,
        data_path: str,
        processor,
        config={}
    ):
        """
        Initialize WanVideo dataset.
        
        Args:
            data_path: Path to JSONL or CSV metadata file
            processor: WanVideoDataProcessor instance
            frame_num: Number of frames to sample from each video
            video_backend: Backend for video loading ('qwen_vl_utils' or 'decord')
        """
        super().__init__(config)
        self.config = config
        self.data_path = Path(data_path)
        self.processor = processor
        
        # Load metadata
        self.samples = self._load_metadata()
        
        # Initialize valid_indices after loading samples
        self.valid_indices = [i for i in range(len(self.samples)) if i not in self.skip]
        
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from JSONL or CSV file."""
        samples = []
        
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r') as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                samples = json.load(f)
        elif self.data_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(self.data_path)
            samples = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
        return samples
    
    def _load_video_frames(self, video_path: str, data_folder=None, fps: int = 1) -> Tuple[np.ndarray, float]:
        """Load video frames using the specified backend."""
        if self.config.data_folder is not None:
            video_path = os.path.join(self.config.data_folder, video_path)

        if self.config.video_backend == "decord":
            return self.load_video_decord(video_path, fps)
        elif self.config.video_backend == "qwen_vl_utils":
            return self.load_video_qwen_vl_utils(video_path, fps)
        else:
            raise ValueError(f"Unsupported video backend: {self.config.video_backend}")

    def load_video_decord(
        self,
        video_path: Union[str, List[str], BytesIO],
        fps: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Load video using Decord backend.

        Args:
            video_path: Path to video file or BytesIO object
            fps: Target frames per second

        Returns:
            Tuple of (video frames, sample fps)
        """
        print(f"{video_path=}", flush=True)

        if isinstance(video_path, str) or isinstance(video_path, BytesIO):
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        elif isinstance(video_path, list):
            vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
        else:
            raise ValueError(f"Unsupported video path type: {type(video_path)}")

        total_frames, video_fps = len(vr), vr.get_avg_fps()
        if self.config.video_sampling_strategy == "fps":
            nframes = smart_nframes(total_frames, video_fps=video_fps, fps=fps)
        elif self.config.video_sampling_strategy == "frame_num":
            nframes = self.config.frame_num
        else:
            raise ValueError(f"Invalid video sampling strategy: {self.config.video_sampling_strategy}")
        uniform_sampled_frames = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        spare_frames = torch.tensor(spare_frames).permute(0, 3, 1, 2)  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        return spare_frames, sample_fps  # (frames, height, width, channels)

    def load_video_qwen_vl_utils(
        self,
        video_path: str,
        fps: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Load video using Qwen VL utils.

        Args:
            video_path: Path to video file
            fps: Target frames per second

        Returns:
            Tuple of (video frames, sample fps)
        """
        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            "min_frames": 1,
            "max_pixels": self.config.video_max_pixels,
            "max_frames": self.config.video_max_frames,
            "min_pixels": self.config.video_min_pixels,
        }
        print(f"{video_dict=}", flush=True)

        if self.config.video_sampling_strategy == "frame_num":
            is_even = self.config.frame_num % 2 == 0
            n_frames = self.config.frame_num if is_even else self.config.frame_num + 1
            video_dict["nframes"] = n_frames
            frames, sample_fps = fetch_video(video_dict, return_video_sample_fps=True)
            frames = frames.numpy()
            if is_even:
                return frames, sample_fps
            else:
                return frames[:-1], sample_fps
        elif self.config.video_sampling_strategy == "fps":
            video_dict["fps"] = fps
            frames, sample_fps = fetch_video(video_dict, return_video_sample_fps=True)
            frames = frames.numpy()
            return frames, sample_fps
        else:
            raise ValueError(f"Invalid video sampling strategy: {self.config.video_sampling_strategy}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        real_idx = self.valid_indices[idx]
        print(f"{idx=} {real_idx=}", flush=True)
        sample = self.samples[real_idx]
        
        # Load video frames
        video_path = sample['video']
        video_frames, fps = self._load_video_frames(video_path)
        
        # Get prompt
        prompt = sample.get('prompt', '')
        
        # Format as hf_messages (expected by processor)
        hf_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process with processor
        processed = self.processor.process(
            images=None,
            hf_messages=hf_messages,
            videos=[video_frames]
        )
        
        return processed

    def get_collator(self):
        return VisionCollator(self.processor)