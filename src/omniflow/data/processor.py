"""
Processes video and text data for WanVideo training.
"""

from typing import Any, Dict, List, Optional

import torch
from PIL import Image


class WanVideoDataProcessor:
    """Standalone data processor for WanVideo training."""
    
    def __init__(self, config, model_id=None):
        self.config = config
        self.model_id = model_id
        self.processor = None
        self.tokenizer = None

    def apply_prompt_template(self, hf_messages: str) -> str:
        """Apply prompt template for WanVideo."""
        # WanVideo uses direct prompts without special formatting
        prompt = hf_messages[0]["content"][1]["text"]
        return prompt

    def save_pretrained(self, save_directory: str):
        pass

    def build(self):
        """Initialize the processor and tokenizer."""
        from models import WanVideoProcessor as WanVideoModelProcessor
        
        wanvideo_kwargs = self.config.get('extra_kwargs', {})
        self.processor = WanVideoModelProcessor(**wanvideo_kwargs)
        self.tokenizer = self.processor.tokenizer

    def process(self, images: List[Image.Image], hf_messages, videos=None, **kwargs) -> Dict[str, Any]:
        """
        Process a single sample for WanVideo training.

        Args:
            images: List of images (for I2V mode)
            hf_messages: Text prompt/caption for the video
            videos: List of video frames
            video_kwargs: Additional video parameters (fps, etc.)

        Returns:
            Dictionary with processed inputs for training
        """
        if hf_messages is None:
            hf_messages = ""

        # Apply prompt template
        formatted_prompt = self.apply_prompt_template(hf_messages)

        # Process text
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
        else:
            # Dummy text inputs if no tokenizer
            text_inputs = {
                "input_ids": torch.zeros((1, 256), dtype=torch.long),
                "attention_mask": torch.ones((1, 256), dtype=torch.long),
            }

        # Process video frames
        if videos is not None and len(videos) > 0:
            # Videos is a list of frame lists
            video_frames = videos[0] if isinstance(videos[0], list) else videos
            # Process frames using the image processor
            video_inputs = self.processor.image_processor.preprocess(
                video_frames,
                return_tensors="pt",
            )
            pixel_values = video_inputs["pixel_values"]
        else:
            raise ValueError("No video frames provided")

        output = {
            "video": pixel_values.squeeze(0),  # T, H, W, C
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }
        return output
