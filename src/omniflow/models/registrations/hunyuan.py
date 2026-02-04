"""Register a minimal HunyuanVideo model builder.

goal:
- Provide an end-to-end runnable model adapter for OmniFlow trainers.
- Keep it lightweight (no dependency on HunyuanVideo-1.5 code yet).
- Match the common trainer contract:
    outputs = model(batch_dict, scheduler) -> {"loss": Tensor}
- Expose `dit` so existing save logic (save DiT weights) works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from omniflow.registry import register_model
from omniflow.utils.train_utils import count_parameters


@dataclass
class HunyuanConfigShim:
    raw: dict

    def save_pretrained(self, save_directory: str):
        # Best-effort minimal JSON for reproducibility/debug.
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "hunyuan_config.json"), "w") as f:
            json.dump(self.raw, f, indent=2, sort_keys=True)


class HunyuanForTraining(nn.Module):
    """
    Minimal training adapter.

    Accepts batch outputs from `HunyuanVideoDataProcessor.prepare_batch`:
      - pixel_values: Tensor [B,C,T,H,W]
      - (optional) input_ids/attention_mask, text, latents, data_type
    """

    def __init__(self, *, in_channels: int = 3, hidden: int = 32, trainable_modules: str = "dit", raw_config: Optional[dict] = None):
        super().__init__()
        self.trainable_modules = trainable_modules

        # Dummy "DiT" placeholder: small Conv3D stack to ensure we have parameters
        # and can validate the training loop plumbing.
        self.dit = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, in_channels, kernel_size=3, padding=1),
        )

        self.config = HunyuanConfigShim(raw=raw_config or {})

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def freeze_except(self):
        # Keep parity with OmniFlow behavior: default train only `dit`.
        for p in self.parameters():
            p.requires_grad_(False)
        mode = (self.trainable_modules or "dit").lower()
        if mode in ("dit", "all"):
            for p in self.dit.parameters():
                p.requires_grad_(True)
        if mode == "all":
            for p in self.parameters():
                p.requires_grad_(True)

    def forward(self, *args, **kwargs):
        # Match OmniFlow trainer call convention: model(batch_dict, scheduler)
        if len(args) >= 1 and isinstance(args[0], dict):
            batch = args[0]
            return self.forward_train(batch)
        raise TypeError("HunyuanForTraining.forward expects (batch_dict, scheduler)")

    def forward_train(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pixel_values = batch.get("pixel_values", None)
        if pixel_values is None:
            # allow wan-like key for experimentation
            pixel_values = batch.get("video", None)
        if pixel_values is None:
            raise ValueError("Batch must contain 'pixel_values' (or 'video' for compatibility)")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
            raise ValueError(f"Expected pixel_values as [B,C,T,H,W] tensor, got {type(pixel_values)} shape={getattr(pixel_values, 'shape', None)}")

        # Simple denoising-style objective placeholder: predict zeros / identity residual.
        # This is NOT the real Hunyuan loss; it's only to validate the OmniFlow plumbing.
        pred = self.dit(pixel_values.to(dtype=self.dtype))
        loss = F.mse_loss(pred.float(), torch.zeros_like(pred).float(), reduction="mean")
        return {"loss": loss}


@register_model("hunyuan")
def build_hunyuan_model(model_config: dict):
    """
    YAML compatibility (minimal):
      model_config:
        name: hunyuan
        config:
          in_channels: 3
          hidden: 32
          trainable_modules: dit
    """
    cfg = dict(model_config.get("config", {}) or {})
    in_channels = int(cfg.get("in_channels", 3))
    hidden = int(cfg.get("hidden", 32))
    trainable_modules = str(cfg.get("trainable_modules", "dit"))

    model = HunyuanForTraining(
        in_channels=in_channels,
        hidden=hidden,
        trainable_modules=trainable_modules,
        raw_config={"model_config": model_config, "resolved": cfg},
    )

    total_params, trainable_params = count_parameters(model)
    logger.info(
        f"hunyuan(dummy) parameters: total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M"
    )
    # Match OmniFlow convention: freeze on build so trainers see correct trainables.
    model.freeze_except()
    total_params, trainable_params = count_parameters(model)
    logger.info(
        f"hunyuan(dummy) after freeze: total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M"
    )
    return model