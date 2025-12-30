import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger
from safetensors.torch import load_file
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_wanvideo import WanVideoConfig
from .wan_video_dit import WanDitModel, sinusoidal_embedding_1d
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE38, WanVideoVAE
from ..debug_utils import print_tensor

PATTERN = "B C H W"


@dataclass
class WanVideoOutput(ModelOutput):
    noise_pred: Optional[torch.FloatTensor] = None
    text_embeddings: Optional[torch.FloatTensor] = None


class WanVideoPreTrainedModel(PreTrainedModel):
    config: WanVideoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["DiTBlock"]
    # _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class WanVideoForConditionalGeneration(WanVideoPreTrainedModel):
    def __init__(self, config: WanVideoConfig):
        super().__init__(config)
        self.config = config

        # Main DiT model
        # self.dit = WanDitModel(config)

        if config.vae_type == "wan_video_vae_38":
            self.vae = WanVideoVAE38()
        elif config.vae_type == "wan_video_vae":
            self.vae = WanVideoVAE()
        else:
            raise ValueError(f"Unsupported vae_type: {config.vae_type}")
            
        self.text_encoder = WanTextEncoder()

        self.image_encoder = None

        self.seperated_timestep = config.seperated_timestep
        self.require_vae_embedding = config.require_vae_embedding
        self.require_clip_embedding = config.require_clip_embedding
        self.fuse_vae_embedding_in_latents = config.fuse_vae_embedding_in_latents

        # The following parameters are used for shape check.
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.time_division_factor = 4
        self.time_division_remainder = 1
        self.trainable_modules = config.trainable_modules

    def freeze_except(self):
        trainable_modules = [] if not self.trainable_modules else self.trainable_modules.split(",")
        for name, model in self.named_children():
            if name in trainable_modules:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)

    def encode_prompt(self, input_ids, attetnion_mask, device="cuda"):
        seq_lens = attetnion_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attetnion_mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def preprocess_video(
        self,
        video,
        dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        # Support both list of frames (single video) and batch tensor inputs
        if isinstance(video, torch.Tensor) and video.ndim == 5:
            # Assume input shape is (B, C, T, H, W)
            return video
        # Original behavior for list of PIL images or tensors per frame
        video = [repeat(image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {})) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_dtype=torch.float32,
        device=None,
        dtype=None,
    ):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_dtype)
        noise = noise.to(dtype=dtype or self.dtype, device=device or self.device)
        return noise

    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            )
            logger.info(f"height % {self.height_division_factor} != 0. We round it up to {height}.")

        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            logger.info(f"width % {self.width_division_factor} != 0. We round it up to {width}.")

        if num_frames is not None:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    num_frames + self.time_division_factor - 1
                ) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                logger.info(
                    f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}."
                )

        return height, width, num_frames

    def noise_initialize(self, height, width, num_frames, seed, rand_device, vace_reference_image, batch_size=1):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        shape = (
            batch_size,
            self.vae.model.z_dim,
            length,
            height // self.vae.upsampling_factor,
            width // self.vae.upsampling_factor,
        )
        noise = self.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return noise

    def embed_input_video(self, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        input_video = self.preprocess_video(input_video)  # B, C, T, H, W
        input_latents = self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=self.dtype, device=self.device)
        if vace_reference_image is not None:
            vace_reference_image = self.preprocess_video([vace_reference_image])
            vace_reference_latents = self.vae.encode(vace_reference_image, device=self.device).to(
                dtype=self.dtype, device=self.device
            )
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        return input_latents


    def forward_preprocess(self, scheduler, data_inputs: dict[str, Any]):
        inputs = data_inputs
        height, width, num_frames = self.check_resize_height_width(
            inputs["height"], inputs["width"], inputs["num_frames"]
        )
        inputs.update({"height": height, "width": width, "num_frames": num_frames})

        if inputs.get("video") is not None and isinstance(inputs["video"], torch.Tensor):
            vid = inputs["video"]
            if vid.ndim == 5:
                # (B, C, F, H, W)
                if vid.shape[2] != num_frames or vid.shape[3] != height or vid.shape[4] != width:
                    # vid.shape[2:] is (F, H, W)
                    logger.info(f"Resizing input video tensor (5D) from {vid.shape[2:]} to {(num_frames, height, width)}")
                    vid = torch.nn.functional.interpolate(vid, size=(num_frames, height, width), mode='trilinear', align_corners=False)
                    inputs["video"] = vid
            elif vid.ndim == 4:
                # (F, H, W, C)
                if vid.shape[0] != num_frames or vid.shape[1] != height or vid.shape[2] != width:
                    logger.info(f"Resizing input video tensor (4D) from {vid.shape[:3]} to {(num_frames, height, width)}")
                    vid = vid.permute(3, 0, 1, 2).unsqueeze(0) # (1, C, F, H, W)
                    vid = torch.nn.functional.interpolate(vid, size=(num_frames, height, width), mode='trilinear', align_corners=False)
                    inputs["video"] = vid.squeeze(0).permute(1, 2, 3, 0).contiguous()
        batch_size = 1
        if inputs.get("video") is not None and isinstance(inputs["video"], torch.Tensor) and inputs["video"].ndim == 5:
             batch_size = inputs["video"].shape[0]
        elif inputs.get("input_ids") is not None:
             batch_size = inputs["input_ids"].shape[0]
            
        noise = self.noise_initialize(
            inputs["height"],
            inputs["width"],
            inputs["num_frames"],
            inputs["seed"],
            self.device,
            inputs["vace_reference_image"],
            batch_size=batch_size,
        )
        inputs.update({"noise": noise})

        if inputs["video"] is not None:
            input_latents = self.embed_input_video(
                inputs["video"],
                noise,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
                inputs["vace_reference_image"],
            )
            if not scheduler.training:
                latents = scheduler.add_noise(input_latents, noise, timestep=scheduler.timesteps[0])
                inputs.update({"latents": latents})
            else:
                inputs.update(
                    {"latents": noise, "input_latents": input_latents}
                )  # this 'latents' actually will not be used in training.
        else:
            inputs.update({"latents": noise})
        # might need to be checked.
        context = self.encode_prompt(inputs["input_ids"], inputs["attention_mask"], device=self.device)
        inputs.update({"context": context})

        return inputs

    def forward(
        self,
        latents,
        context,
        timestep,
        y: Optional[torch.FloatTensor] = None,
        reference_latents: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.FloatTensor] = None,
        vace_context: Optional[torch.FloatTensor] = None,
        vace_scale: Optional[float] = 1.0,
        motion_bucket_id: Optional[int] = None,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> WanVideoOutput:
        x = self.dit(latents, timestep, context)
        return WanVideoOutput(
            noise_pred=x,
            text_embeddings=context,
        )

    def load_model(self, vae_ckpt_path, text_encoder_ckpt_path):
        # Load pretrained weights for frozen components
        if vae_ckpt_path:
            print(f"Loading VAE from {vae_ckpt_path}")
            vae_state_dict = torch.load(
                vae_ckpt_path, map_location="cpu"
            )
            
            # Check if we need to add 'model.' prefix
            if "model.encoder.conv1.weight" not in vae_state_dict and "encoder.conv1.weight" in vae_state_dict:
                print("Detected missing 'model.' prefix in VAE checkpoint. Adding it...")
                new_vae_state_dict = {}
                for k, v in vae_state_dict.items():
                    new_vae_state_dict[f"model.{k}"] = v
                vae_state_dict = new_vae_state_dict

            self.vae.load_state_dict(vae_state_dict, strict=True, assign=True)
            print("VAE loaded.")

        if text_encoder_ckpt_path:
            print(f"Loading T5 from {text_encoder_ckpt_path}")
            t5_state_dict = torch.load(
                text_encoder_ckpt_path, map_location="cpu"
            )
            self.text_encoder.load_state_dict(t5_state_dict, strict=True, assign=True)
            print("T5 loaded.")

__all__ = [
    "WanVideoForConditionalGeneration",
    "WanVideoPreTrainedModel",
]
