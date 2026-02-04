"""
HunyuanVideo data processor (batch preparer).
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch


class HunyuanVideoDataProcessor:
    """
    A minimal batch preparer for HunyuanVideo-style training.

    Expected raw sample keys (dataset may provide any subset):
      - text or prompt: str
      - pixel_values: torch.Tensor or np.ndarray
          video: [C,T,H,W] or [T,H,W,C] or already batched [B,C,T,H,W]
          image: [C,H,W] or [H,W,C]
      - data_type: str (optional, e.g. "t2v"/"i2v"/"video"/"image")
      - latents: torch.Tensor (optional precomputed VAE latents)

    Output:
      - pixel_values: torch.Tensor [B,C,T,H,W]
      - text: list[str]
      - (optional) input_ids/attention_mask if text_tokenizer configured
      - (optional) data_type/latents passthrough
    """

    def __init__(self, config: dict, model_id: Optional[str] = None):
        self.config = config or {}
        self.model_id = model_id
        self.tokenizer = None

    def build(self):
        # Idempotent build
        if self.tokenizer is not None:
            return
        tok = self.config.get("text_tokenizer")
        if not tok:
            return
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tok)

    def _normalize_raw_batch(
        self, batch: Any
    ) -> Tuple[list[str], list[Any], Optional[list[Any]], Optional[list[str]]]:
        """
        Normalize dataloader output to:
          - texts: list[str]
          - pixel_values_list: list[Any] (each item is a single sample's pixels)
          - latents_list: Optional[list[Any]]
          - data_types: Optional[list[str]]
        """
        if isinstance(batch, dict):
            texts = batch.get("text", batch.get("prompt"))
            pixel_values_list = batch.get("pixel_values", batch.get("video_frames", batch.get("video")))
            latents_list = batch.get("latents", None)
            data_types = batch.get("data_type", None)

            if texts is None or pixel_values_list is None:
                raise ValueError(
                    "prepare_batch(dict) requires keys: ('text' or 'prompt') and "
                    "('pixel_values' or 'video_frames' or 'video')"
                )

            if isinstance(texts, str):
                texts = [texts]
            if isinstance(data_types, str):
                data_types = [data_types] * len(texts)

            if not isinstance(texts, (list, tuple)):
                raise TypeError(f"'text/prompt' must be str or list[str], got {type(texts)}")
            if not isinstance(pixel_values_list, (list, tuple)):
                # Allow already-batched tensor/ndarray
                pixel_values_list = [pixel_values_list]

            if latents_list is not None and not isinstance(latents_list, (list, tuple)):
                latents_list = [latents_list]

            return list(map(str, texts)), list(pixel_values_list), list(latents_list) if latents_list is not None else None, list(data_types) if data_types is not None else None

        # RawBatchCollator returns list[dict]
        if not isinstance(batch, (list, tuple)) or not batch:
            raise ValueError(f"prepare_batch expected non-empty list/tuple, got {type(batch)}")

        texts: list[str] = []
        pixels: list[Any] = []
        latents: list[Any] = []
        data_types: list[str] = []
        has_latents = False
        has_data_type = False

        for ex in batch:
            texts.append(str(ex.get("text", ex.get("prompt", ""))))
            pixels.append(ex.get("pixel_values", ex.get("video_frames", ex.get("video"))))

            if "latents" in ex and ex.get("latents") is not None:
                has_latents = True
                latents.append(ex.get("latents"))
            elif has_latents:
                latents.append(None)

            if "data_type" in ex and ex.get("data_type") is not None:
                has_data_type = True
                data_types.append(str(ex.get("data_type")))
            elif has_data_type:
                data_types.append("unknown")

        return texts, pixels, latents if has_latents else None, data_types if has_data_type else None

    @staticmethod
    def _as_torch(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        # numpy arrays / array-likes
        try:
            import numpy as np  # local import

            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
        except Exception:
            pass
        raise TypeError(f"Unsupported pixel/latent type: {type(x)}")

    @staticmethod
    def _to_bcthw(pixel: torch.Tensor) -> torch.Tensor:
        """
        Convert a single-sample pixel tensor to [C,T,H,W].
        Accepts common layouts:
          - [T,H,W,C] (HWC frames)
          - [C,T,H,W]
          - [H,W,C] (single image)
          - [C,H,W] (single image)
        """
        if pixel.ndim == 4:
            # video: [T,H,W,C] or [C,T,H,W]
            if pixel.shape[-1] in (1, 3, 4):
                # [T,H,W,C] -> [C,T,H,W]
                return pixel.permute(3, 0, 1, 2).contiguous()
            return pixel.contiguous()  # assume [C,T,H,W]
        if pixel.ndim == 3:
            # image: [H,W,C] or [C,H,W]
            if pixel.shape[-1] in (1, 3, 4):
                pixel = pixel.permute(2, 0, 1)
            return pixel.unsqueeze(1).contiguous()  # [C,1,H,W]
        raise ValueError(f"Unsupported pixel ndim={pixel.ndim}, shape={tuple(pixel.shape)}")

    def prepare_batch(self, *, batch: Any, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        # v0: only normalize + optional tokenize; keep heavy encoding in model/pipeline.
        if self.tokenizer is None and self.config.get("text_tokenizer"):
            self.build()

        texts, pixels_list, latents_list, data_types = self._normalize_raw_batch(batch)

        # Pixel values: convert to [B,C,T,H,W]
        # Allow already-batched [B,C,T,H,W]
        if len(pixels_list) == 1:
            px0 = self._as_torch(pixels_list[0])
            if px0.ndim == 5:
                pixel_values = px0
            else:
                pixel_values = self._to_bcthw(px0).unsqueeze(0)
        else:
            per_sample = [self._to_bcthw(self._as_torch(p)) for p in pixels_list]
            pixel_values = torch.stack(per_sample, dim=0)

        # Basic dtype normalization (do not move device here; trainers handle that)
        if pixel_values.dtype == torch.uint8:
            pixel_values = pixel_values.to(torch.float32).div(255.0)
        # honor requested dtype for downstream model
        if dtype is not None and pixel_values.dtype != dtype and pixel_values.is_floating_point():
            pixel_values = pixel_values.to(dtype=dtype)

        out: Dict[str, Any] = {
            "pixel_values": pixel_values,  # [B,C,T,H,W]
            "text": texts,
        }

        if data_types is not None:
            out["data_type"] = data_types

        if latents_list is not None:
            # If latents are provided, stack where possible; keep None if missing.
            lat_tensors: list[torch.Tensor] = []
            for x in latents_list:
                if x is None:
                    raise ValueError("latents_list contains None; provide latents for all samples or omit key")
                lat_tensors.append(self._as_torch(x))
            out["latents"] = torch.stack(lat_tensors, dim=0)

        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=self.config.get("padding_strategy", "max_length"),
                truncation=True,
                max_length=self.config.get("max_text_length", 512),
            )
            out.update(text_inputs)

        return out