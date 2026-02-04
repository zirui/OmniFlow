import os
from dataclasses import dataclass
from typing import Any, Optional 

import torch
import torch.nn as nn

from einops import repeat
from loguru import logger
import json
import glob
from safetensors.torch import load_file
import torch.distributed as dist

from ..interface import GenAIModel
from .configuration_wanvideo import WanVideoConfig
from .wan_video_dit import WanDitModel, sinusoidal_embedding_1d
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE38, WanVideoVAE

PATTERN = "B C H W"


@dataclass
class WanVideoOutput:
    noise_pred: Optional[torch.FloatTensor] = None
    text_embeddings: Optional[torch.FloatTensor] = None


class WanVideoForConditionalGeneration(GenAIModel, nn.Module):
    def __init__(self, config: WanVideoConfig):
        super().__init__()
        self.config = config
        # Main DiT model
        self.dit = WanDitModel(config)
        if config.vae_type == "wan_video_vae_38":
            self.vae = WanVideoVAE38()
        elif config.vae_type == "wan_video_vae":
            self.vae = WanVideoVAE()
        else:
            raise ValueError(f"Unsupported vae_type: {config.vae_type}")
            
        self.text_encoder = WanTextEncoder()
        
        self.is_gradient_checkpointing = False

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

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save state dict
        model_path = os.path.join(save_directory, "diffusion_pytorch_model.safetensors")
        from safetensors.torch import save_file
        save_file(self.state_dict(), model_path)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.is_gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False

    def freeze_except(self):
        trainable_modules = [] if not self.trainable_modules else self.trainable_modules.split(",")
        for name, model in self.named_children():
            if name in trainable_modules:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)

    def encode_prompt(self, input_ids, attention_mask, device="cuda"):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
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

    def noise_initialize(self, height, width, num_frames, seed, rand_device, batch_size=1):
        length = (num_frames - 1) // 4 + 1
        shape = (
            batch_size,
            self.vae.model.z_dim,
            length,
            height // self.vae.upsampling_factor,
            width // self.vae.upsampling_factor,
        )
        noise = self.generate_noise(shape, seed=seed, rand_device=rand_device)
        return noise

    def embed_input_video(self, input_video, noise, tiled, tile_size, tile_stride):
        input_video = self.preprocess_video(input_video)  # B, C, T, H, W
        input_latents = self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=self.dtype, device=self.device)
        return input_latents

    def embed_image_VAE(
        self,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        image = repeat(input_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {}))
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = repeat(end_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {}))
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = self.vae.encode(
            [vae_input.to(dtype=self.dtype, device=self.device)],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=self.dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=self.dtype, device=self.device)
        return y

    def embed_image_CLIP(self, input_image, end_image, height, width):
        if input_image.ndim == 4:  # (B, C, H, W)
            image = input_image
            # Assuming image_encoder.encode_image can handle batched tensor input
            # If it expects list, we might need to adjust, but usually it's fine.
            # WanTextEncoder/CLIP wrapper usually handles tensor.
            clip_context = self.image_encoder.encode_image(image)
        else:
            image = repeat(input_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {}))
            clip_context = self.image_encoder.encode_image([image])

        if end_image is not None:
            if end_image.ndim == 4:
                end_image_processed = end_image
                end_feat = self.image_encoder.encode_image(end_image_processed)
            else:
                end_image_processed = repeat(end_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {}))
                end_feat = self.image_encoder.encode_image([end_image_processed])
            
            if self.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, end_feat], dim=1)
                
        clip_context = clip_context.to(dtype=self.dtype, device=self.device)
        return clip_context

    def embed_image_fused(self, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image.ndim == 4: # (B, C, H, W)
            image = repeat(input_image, "b c h w -> b c t h w", t=1)
            # Pass directly as 'videos' (tensor) to vae.encode
            z = self.vae.encode(
                image,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        else:
            image = repeat(input_image, f"H W C -> C T H W", T=1)
            z = self.vae.encode(
                [image],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        
        latents[:, :, 0:1] = z
        return latents, z

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
                # logger.debug(f"DEBUG: video input shape: {vid.shape}")

                # Ensure dtype matches model
                if vid.dtype != self.dtype:
                    logger.info(f"Casting video from {vid.dtype} to {self.dtype}")
                    vid = vid.to(self.dtype)

                inputs["video"] = vid
        batch_size = 1
        if inputs.get("video") is not None and isinstance(inputs["video"], torch.Tensor) and inputs["video"].ndim == 5:
             batch_size = inputs["video"].shape[0]
        elif inputs.get("input_ids") is not None:
             batch_size = inputs["input_ids"].shape[0]

        # Force eval mode for frozen encoders to disable dropout
        self.text_encoder.eval()
        self.vae.eval()

        noise = self.noise_initialize(
            inputs["height"],
            inputs["width"],
            inputs["num_frames"],
            inputs["seed"],
            "cpu",
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
        context = context.to(self.dit.dtype)
        inputs.update({"context": context})

        # logger.info(f"OmniFlow Context Mean: {context.mean().item():.6f}, Std: {context.std().item():.6f}")

        if inputs["input_image"] is not None and self.require_vae_embedding:
            y = self.embed_image_VAE(
                inputs["input_image"],
                inputs["end_image"],
                num_frames,
                height,
                width,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
            )
            inputs.update({"y": y})

        if (inputs["input_image"] is not None) and (self.image_encoder is not None) and self.require_clip_embedding:
            clip_feature = self.embed_image_CLIP(inputs["input_image"], inputs["end_image"], height, width)
            inputs.update({"clip_feature": clip_feature})

        if inputs["input_image"] is not None and self.fuse_vae_embedding_in_latents:
            latents, first_frame_latents = self.embed_image_fused(
                inputs["input_image"],
                inputs["latents"],
                height,
                width,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
            )
            inputs.update(
                {
                    "latents": latents,
                    "fuse_vae_embedding_in_latents": True,
                    "first_frame_latents": first_frame_latents,
                }
            )

        return inputs

    def forward(self, *args, **kwargs):
        """
        Dispatch method:
        - If input is a dict (batch from implementation), call forward_train.
        - Otherwise call forward_dit (legacy/inference).
        """
        if len(args) > 0 and isinstance(args[0], dict) and "video" in args[0]:
            scheduler = args[1] if len(args) > 1 else kwargs.get("scheduler")
            return self.forward_train(args[0], scheduler)
        return self.forward_dit(*args, **kwargs)

    def forward_train(self, batch: dict[str, Any], scheduler=None) -> dict[str, torch.Tensor]:
        device = self.device
        
        # 1. Prepare Inputs
        pixel_values = batch.get("video")
        if pixel_values is None:
            raise ValueError("Batch must contain 'video' key")

        # Handle tensor movement if needed (though usually done by trainer/dataloader)
        if isinstance(pixel_values, torch.Tensor) and pixel_values.device != device:
             pixel_values = pixel_values.to(device, non_blocking=True)
        
        # Extract dims
        if pixel_values.ndim == 5:
            num_frames, height, width = pixel_values.shape[2:5]
        else:
            num_frames, height, width = pixel_values.shape[:3]

        inputs_dict = {
            "video": pixel_values,
            "input_ids": batch.get("input_ids"),
            "attention_mask": batch.get("attention_mask"),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            # Keep consistent with DiffSynth-Studio TI2V training behavior:
            # TI2V/I2V conditioning is only enabled when the batch explicitly provides `input_image`
            # (e.g. via a dataset field or an "extra_inputs" mechanism in the training script).
            "input_image": batch.get("input_image", None),
            "cfg_scale": batch.get("cfg_scale", 1),
            "cfg_merge": batch.get("cfg_merge", False),
            "seed": batch.get("seed", None),
            "reference_image": batch.get("reference_image", None),
            "tiled": batch.get("tiled", False),
            "tile_size": batch.get("tile_size", None),
            "tile_stride": batch.get("tile_stride", None),
            "end_image": batch.get("end_image", None),
        }

        # 2. Handle Fixed Seed
        if os.environ.get("FIXED_SEED"):
            try:
                fixed_seed = int(os.environ["FIXED_SEED"])
                inputs_dict["seed"] = fixed_seed
            except ValueError:
                raise ValueError(f"Invalid FIXED_SEED value: {os.environ['FIXED_SEED']}")

        # 3. Handle Timestep Selection
        if os.environ.get("FIXED_TIMESTEP"):
            try:
                fixed_step = int(os.environ["FIXED_TIMESTEP"])
                max_step = scheduler.num_train_timesteps - 1
                if fixed_step < 0 or fixed_step > max_step:
                    logger.warning(f"FIXED_TIMESTEP {fixed_step} out of range [0, {max_step}]. Clamping.")
                    fixed_step = max(0, min(fixed_step, max_step))
                
                timestep_id = torch.tensor([fixed_step], device=device)
            except ValueError:
                raise ValueError(f"Invalid FIXED_TIMESTEP value: {os.environ['FIXED_TIMESTEP']}")
        else:
            max_timestep_boundary = int(1 * scheduler.num_train_timesteps)
            min_timestep_boundary = int(0 * scheduler.num_train_timesteps)
            timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,), device=device)

        timestep = scheduler.timesteps[timestep_id.cpu()]

        # 4. Forward Preprocess (VAE, TextEnc, Noise Init)
        # Note: forward_preprocess expects inputs_dict values to be on device if they are tensors
        # We ensure generic tensor movement here for keys that might be tensors
        for k, v in inputs_dict.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                 inputs_dict[k] = v.to(device, non_blocking=True)
        
        with torch.no_grad():
             pre_processed_inputs = self.forward_preprocess(scheduler, inputs_dict)
        timestep = timestep.float()

        # 5. Diffusion Target & Noise Injection
        training_target = scheduler.training_target(
            pre_processed_inputs["input_latents"],
            pre_processed_inputs["noise"],
            timestep,
        )
        pre_processed_inputs["latents"] = scheduler.add_noise(
            pre_processed_inputs["input_latents"],
            pre_processed_inputs["noise"],
            timestep,
        )

        # 6. DiT Forward
        output = self.forward_dit(
            latents=pre_processed_inputs.get("latents", None),
            context=pre_processed_inputs.get("context", None),
            timestep=timestep,
            y=pre_processed_inputs.get("y", None),
            reference_latents=pre_processed_inputs.get("reference_latents", None),
            clip_feature=pre_processed_inputs.get("clip_feature", None),
            fuse_vae_embedding_in_latents=bool(pre_processed_inputs.get("fuse_vae_embedding_in_latents", False)),
        )

        noise_pred = output.noise_pred
        
        # 7. Loss Calculation
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="mean")
        weight = scheduler.training_weight(timestep)
        loss = loss * weight

        return {"loss": loss}

    def forward_inference(self, batch: dict[str, Any], **kwargs):
        pass

    def forward_dit(
        self,
        latents,
        context,
        timestep,
        y: Optional[torch.FloatTensor] = None,
        reference_latents: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.FloatTensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> WanVideoOutput:
        # TODO: zirui, for debugging
        # print(f"DEBUG:  {self.seperated_timestep=}, {self.config.fuse_vae_embedding_in_latents=}")
        # DiffSynth-Studio gates separated timestep on runtime fuse flag.
        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            # Logic to split timestep for T2V (first frame t=0) to match DiffSynth/WanVideo
            # if torch.rand(1).item() < 0.001: 
            #      logger.info("DEBUG: Using separated_timestep logic (first frame t=0)")
            F = latents.shape[2]
            H = latents.shape[3]
            W = latents.shape[4]
            spatial_size = (H * W) // 4
            
            # Ensure timestep is (B,)
            if timestep.ndim == 0:
                timestep = timestep.unsqueeze(0).repeat(latents.shape[0])
            elif timestep.shape[0] != latents.shape[0]:
                timestep = timestep.repeat(latents.shape[0])
            
            # t_seq: (B, F, spatial_patches)
            time_seq = torch.zeros((latents.shape[0], F, spatial_size), dtype=timestep.dtype, device=timestep.device)
            # Fill frames 1..F with timestep
            time_seq[:, 1:, :] = timestep.view(-1, 1, 1)
            
            # Flatten to (B, L)
            timestep_input = time_seq.flatten(1)
            
            # Embed
            # sinusoidal_embedding_1d expects 1D input, so we flatten B*L
            flat_timestep = timestep_input.flatten()
            emb = sinusoidal_embedding_1d(self.dit.freq_dim, flat_timestep)
            # Reshape back to (B, L, D) - actually embedding layer might handle it but let's be safe
            emb = emb.view(latents.shape[0], -1, self.dit.freq_dim)
            
            t = self.dit.time_embedding(emb.to(device=self.dit.device, dtype=self.dit.dtype))
            
            # Projection to (B, L, 6, H) - Note unflatten dim 2
            t_mod = self.dit.time_projection(t).unflatten(2, (6, self.dit.hidden_size))
            
        else:
            t = self.dit.time_embedding(
                sinusoidal_embedding_1d(self.dit.freq_dim, timestep).to(device=self.dit.device, dtype=self.dit.dtype)
            )
            t_mod = self.dit.time_projection(t).unflatten(1, (6, self.dit.hidden_size))

        context = self.dit.text_embedding(context)

        x = latents
        # Merged cfg
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        # Image Embedding
        if y is not None and self.require_vae_embedding:
            x = torch.cat([x, y], dim=1)

        if clip_feature is not None and self.require_clip_embedding:
            clip_embdding = self.dit.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        # Add camera control(TODO: remove)
        x, (f, h, w) = self.dit.patchify(x)

        # Reference image
        if reference_latents is not None:
            if len(reference_latents.shape) == 5:  # video case
                reference_latents = reference_latents[:, :, 0]  # only use the first frame
            reference_latents = self.dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
            x = torch.concat([reference_latents, x], dim=1)
            f += 1

        freqs = (
            torch.cat(
                [
                    self.dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )


        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block_id, block in enumerate(self.dit.blocks):
            if self.training and self.is_gradient_checkpointing:
                 x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.dit.head(x, t)
        # Remove reference latents
        if (
            reference_latents is not None
        ):  # since we replace the first frame with the reference image, we need to remove the first frame
            x = x[:, reference_latents.shape[1] :]
            f -= 1
        x = self.dit.unpatchify(x, (f, h, w))

        return WanVideoOutput(
            noise_pred=x,
            text_embeddings=context,
        )

    @classmethod
    def load_dit(cls, pretrained_path, device="cpu", dtype=None, **kwargs):
        """
        Load a DiT-only checkpoint from a directory or file.
        Arguments:
            pretrained_path: Path to the directory containing config.json and diffusion_pytorch_model.safetensors, 
                           or path to the safetensors file directly (assuming config.json is in the same dir).
            device: Device to load the model to.
            dtype: Dtype to load the model to.
            kwargs: Additional arguments to override the config (e.g. vae_type, trainable_modules).
        """
        
        pretrained_path = str(pretrained_path)
        weight_files = []
        
        if os.path.isfile(pretrained_path):
            if pretrained_path.endswith(".safetensors") or pretrained_path.endswith(".bin"):
                weight_files = [pretrained_path]
                config_path = os.path.join(os.path.dirname(pretrained_path), "config.json")
            else:
                 raise ValueError(f"Unsupported file type: {pretrained_path}")
        else:
            # We look for files matching *model*.safetensors or *model*.bin in directory
            safetensor_files = sorted(glob.glob(os.path.join(pretrained_path, "*model*.safetensors")))
            if safetensor_files:
                weight_files = safetensor_files
            else:
                bin_files = sorted(glob.glob(os.path.join(pretrained_path, "*model*.bin")))
                if bin_files:
                    weight_files = bin_files
            
            config_path = os.path.join(pretrained_path, "config.json")
            
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")
            
        if not weight_files:
             raise ValueError(f"No checkpoint files found in {pretrained_path}")

        # 1. Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        # 2. Convert config
        wan_config_kwargs = config_dict.copy()
        
        # Direct mapping attempt
        key_map = {
            "dim": "dit_hidden_size",
            "num_layers": "dit_num_layers",
            "num_heads": "dit_num_heads",
            "ffn_dim": "dit_intermediate_size",
            "in_dim": "dit_in_channels",
            "out_dim": "dit_out_channels",
            "freq_dim": "dit_freq_dim",
            "text_len": "text_len", 
        }
        
        for k, v in config_dict.items():
            if k in key_map:
                wan_config_kwargs[key_map[k]] = v
        
        if "patch_size" in config_dict:
             wan_config_kwargs["dit_patch_size"] = tuple(config_dict["patch_size"])
             
        # Merge kwargs (overrides)
        for k, v in kwargs.items():
            wan_config_kwargs[k] = v

        # Create WanVideoConfig
        config = WanVideoConfig(**wan_config_kwargs)
        
        # 3. Initialize Model
        # logger.info(f"Initializing WanVideoForConditionalGeneration with config: {config}")
        model = cls(config)
        
        # 4. Load Weights (Supports sharded)
        state_dict = {}
        for ckpt_path in weight_files:
            # logger.info(f"Loading weights from {ckpt_path}")
            if ckpt_path.endswith(".safetensors"):
                # logger.info(f"DEBUG: Loading safetensors from {ckpt_path}")
                part_state_dict = load_file(ckpt_path)
            else:
                part_state_dict = torch.load(ckpt_path, map_location="cpu")
            state_dict.update(part_state_dict)
            
        # 5. Key Remapping (blocks. -> dit.blocks.)
        tensors_to_load = {}
        for k, v in state_dict.items():
            if k.startswith("dit."):
                tensors_to_load[k] = v
            # Check for DiT keys
            elif k.startswith('blocks.') or k.startswith('patch_embedding.') or k.startswith('text_embedding.') or k.startswith('time_embedding.') or k.startswith('time_projection.') or k.startswith('head.') or k.startswith('img_emb.') or k.startswith('ref_conv.'):
                 tensors_to_load[f"dit.{k}"] = v
            else:
                 tensors_to_load[k] = v
                 
        # 6. Load into model
        missing, unexpected = model.load_state_dict(tensors_to_load, strict=False)
        logger.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # 7. Move to device/dtype
        if dtype is not None:
             model.to(dtype=dtype)
        if device != "cpu":
             model.to(device)
             
        return model

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        return cls.load_dit(pretrained_path, **kwargs)


__all__ = [
    "WanVideoForConditionalGeneration",
]
