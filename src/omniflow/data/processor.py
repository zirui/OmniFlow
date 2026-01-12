"""
Processes video and text data for training.
"""

from typing import Any, Dict, List 

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
        from transformers import AutoTokenizer
        from models import WanVideoProcessor as WanVideoModelProcessor
        
        wanvideo_kwargs = self.config.get('extra_kwargs', {})

        # Load tokenizer if specified
        if self.config.get('text_tokenizer', None) is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.get('text_tokenizer'))
        else:
            self.tokenizer = None
        self.processor = WanVideoModelProcessor(**wanvideo_kwargs, tokenizer=self.tokenizer)

        if self.tokenizer is None:
            self.tokenizer = self.processor.tokenizer
            # AutoTokenizer.from_pretrained(self.model_id)
            print("use processor's tokenizer")

    def process(self, images: List[Image.Image], hf_messages, videos=None, **kwargs) -> Dict[str, Any]:
        """
        Process a single sample for WanVideo training.

        Args:
            images: List of images (for I2V mode)
            hf_messages: Text prompt/caption for the video
            videos: List of video frames
            kwargs: Additional video parameters (fps, num_frames, etc.)

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
                padding=self.config.get('padding_strategy', "max_length"),
                truncation=True,
                max_length=self.config.get('max_text_length', 512),
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
                num_frames=kwargs.get('num_frames', None),
                return_tensors="pt",
            )
            # TODO: zirui, temporarily convert to bfloat16 for debugging(fix video max discrepancy issue)
            pixel_values = video_inputs["pixel_values"].to(torch.bfloat16)
        else:
            raise ValueError("No video frames provided")

        output = {
            "video": pixel_values.squeeze(0),  # C, T, H, W
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "num_frames": pixel_values.shape[2],  # (1, C, T, H, W) -> T used shape[2]
        }
        return output
