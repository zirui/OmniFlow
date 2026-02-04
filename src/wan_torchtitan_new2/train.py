# Training entry for wan_new2 TorchTitan experiment.

from __future__ import annotations

from typing import Optional, Dict
import os
import glob

import torch

from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

from omniflow.schedulers.flow_match import FlowMatchScheduler
from omniflow.models.wan_new2.t5 import umt5_xxl_encoder_from_checkpoint
from omniflow.models.wan_new2.vae2_1 import Wan2_1_VAE
from omniflow.models.wan_new2.vae2_2 import Wan2_2_VAE
from safetensors.torch import load_file as safe_load_file


def _encode_prompt(text_encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lens = attention_mask.gt(0).sum(dim=1).long()
    prompt_emb = text_encoder(input_ids, attention_mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[i, v:] = 0
    return prompt_emb


def _vae_encode(vae, videos_bcthw: torch.Tensor) -> torch.Tensor:
    # Wan VAEs here take list[[C,T,H,W]].
    videos_list = [videos_bcthw[i] for i in range(videos_bcthw.shape[0])]
    latents_list = vae.encode(videos_list)
    return torch.stack(latents_list)


def _make_weight_and_sigma_tables() -> tuple[torch.Tensor, torch.Tensor]:
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler.sigmas, scheduler.linear_timesteps_weights


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return dict(safe_load_file(path))
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def _find_dit_weight_files(pretrained_path: str) -> list[str]:
    if os.path.isfile(pretrained_path):
        return [pretrained_path]

    preferred = [
        "diffusion_pytorch_model.safetensors",
        "dit_model.safetensors",
        "model.safetensors",
    ]
    candidates: list[str] = []
    for fname in preferred:
        p = os.path.join(pretrained_path, fname)
        if os.path.exists(p):
            candidates.append(p)
    if candidates:
        return candidates

    candidates = sorted(glob.glob(os.path.join(pretrained_path, "*model*.safetensors")))
    if candidates:
        return candidates
    candidates = sorted(glob.glob(os.path.join(pretrained_path, "*model*.bin")))
    if candidates:
        return candidates
    raise FileNotFoundError(f"No DiT weights found under {pretrained_path}")


def _load_dit_weights_into_module(dit: torch.nn.Module, pretrained_path: str) -> None:
    files = _find_dit_weight_files(pretrained_path)
    merged: Dict[str, torch.Tensor] = {}
    for fpath in files:
        merged.update(_strip_module_prefix(_load_state_dict(fpath)))

    if any(k.startswith("dit.") for k in merged.keys()):
        merged = {k[len("dit.") :]: v for k, v in merged.items() if k.startswith("dit.")}

    try:
        param_dtype = next(dit.parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32

    if param_dtype in (torch.float16, torch.bfloat16):
        merged = {
            k: (v.to(dtype=param_dtype) if torch.is_floating_point(v) else v)
            for k, v in merged.items()
        }

    result = dit.load_state_dict(merged, strict=False)
    logger.info(
        f"[wan_new2] Loaded DiT weights from {pretrained_path} (files={len(files)}). "
        f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
    )


class WanNew2Trainer(Trainer):
    """
    TorchTitan trainer that keeps the model as DiT-only,
    and handles VAE/T5 + flow-matching schedule in forward_backward_step.
    """

    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        # Encoders dtype policy:
        # - When using mixed-precision sharding, keep encoders in fp32 (stability + parity with OmniFlow baseline)
        # - Otherwise, keep fp32 too (safe default).
        self._enc_dtype = torch.float32

        # Load encoders (frozen)
        vae_type = getattr(job_config.encoder, "vae_type", "wan2.1")
        if "2.2" in str(vae_type) or "vae_38" in str(vae_type):
            self.vae = Wan2_2_VAE(
                z_dim=48,
                vae_pth=getattr(job_config.encoder, "vae_checkpoint_path", None),
                dtype=self._enc_dtype,
                device=str(self.device),
            )
            self.vae.upsampling_factor = 16
        else:
            self.vae = Wan2_1_VAE(
                z_dim=16,
                vae_pth=getattr(job_config.encoder, "vae_checkpoint_path", None),
                dtype=self._enc_dtype,
                device=str(self.device),
            )
            self.vae.upsampling_factor = 8

        t5_ckpt = getattr(job_config.encoder, "t5_checkpoint_path", None)
        self.text_encoder = umt5_xxl_encoder_from_checkpoint(
            t5_ckpt, dtype=torch.bfloat16, device="cpu"
        ).to(device=self.device, dtype=self._enc_dtype)
        self.text_encoder.eval().requires_grad_(False)

        # Precompute sigma/weight tables (CPU tensors). We'll index by timestep_id and move scalars to CUDA.
        self._sigmas_cpu, self._weights_cpu = _make_weight_and_sigma_tables()
        self._timesteps_cpu = self._sigmas_cpu * 1000.0

        pretrained_path = getattr(job_config.training, "load_from_pretrained_path", "") or ""
        if pretrained_path:
            if self.parallel_dims.dp_shard_enabled:
                logger.warning(
                    "[wan_new2] load_from_pretrained_path is set, but dp_shard is enabled. "
                    "Skipping DiT weight loading for now."
                )
            else:
                m0 = self.model_parts[0]
                inner = m0.module if hasattr(m0, "module") else m0
                dit = getattr(inner, "dit", None)
                if dit is None:
                    raise AttributeError("wan_new2 model must expose `.dit` for weight loading")
                _load_dit_weights_into_module(dit, pretrained_path)

        # IMPORTANT:
        # When FSDP is disabled, TorchTitan keeps model params in fp32 and uses AMP.
        # However, WanNew2 attention backend requires q/k/v projection weights to be
        # half/bf16 (it asserts on `dtype`). Therefore, for AMP-only runs we cast the
        # DiT backbone to bf16 when mixed_precision_param=bfloat16.
        if (
            job_config.training.mixed_precision_param == "bfloat16"
            and not self.parallel_dims.dp_shard_enabled
        ):
            try:
                self.model_parts[0].to(dtype=torch.bfloat16)
                logger.info("[wan_new2] AMP-only: cast DiT params to bf16 for flash-attn backend.")
            except Exception as exc:
                logger.warning(f"[wan_new2] failed to cast DiT params to bf16: {exc}")

        # Optional: align seeds across ranks for debugging (do NOT force deterministic algos by default).
        if os.environ.get("FIXED_SEED"):
            try:
                seed = int(os.environ["FIXED_SEED"])
                dist_utils.set_determinism(
                    self.parallel_dims.world_mesh,
                    self.device,
                    seed,
                    deterministic=bool(getattr(job_config.training, "deterministic", False)),
                )
                logger.info(
                    f"[wan_new2] FIXED_SEED={seed} enabled "
                    f"(deterministic={bool(getattr(job_config.training, 'deterministic', False))})."
                )
            except Exception as exc:
                logger.warning(f"[wan_new2] failed to set FIXED_SEED: {exc}")

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model = self.model_parts[0]  # DiT only (possibly FSDP-sharded)

        video = input_dict.pop("input")
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")

        if input_ids is None or attention_mask is None:
            raise ValueError("wan_new2 requires input_ids and attention_mask from the dataset processor.")

        # Move inputs
        video = video.to(self.device, non_blocking=True)
        # VAE expects float inputs; avoid accidental float64 from preprocessing.
        video = video.to(dtype=self._enc_dtype)
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        # 1) VAE encode (frozen)
        with torch.no_grad():
            latents = _vae_encode(self.vae, video).to(device=self.device)

        # Choose latent dtype for DiT input:
        # - Default: match DiT param dtype (typically bf16 under mixed precision)
        # - Optional: keep fp32 preprocessing for alignment/debug
        model_dtype = next(model.parameters()).dtype
        if os.environ.get("WAN_PREPROCESS_FP32") == "1":
            latents = latents.float()
        else:
            latents = latents.to(dtype=model_dtype)

        # 2) timestep_id + noise
        if os.environ.get("FIXED_TIMESTEP"):
            timestep_id = int(os.environ["FIXED_TIMESTEP"])
            timestep_id = max(0, min(timestep_id, 999))
            timestep_id_t = torch.tensor([timestep_id], device=self.device, dtype=torch.long)
        else:
            timestep_id_t = torch.randint(0, 1000, (1,), device=self.device, dtype=torch.long)
            timestep_id = int(timestep_id_t.item())

        # noise (optional fixed seed, keep parity with existing scripts: noise sampled on CPU)
        if os.environ.get("FIXED_SEED"):
            seed = int(os.environ["FIXED_SEED"])
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(latents.shape, generator=gen, device="cpu", dtype=torch.float32).to(
                device=self.device, dtype=latents.dtype
            )
        else:
            noise = torch.randn_like(latents)

        # 3) schedule by timestep_id (no argmin / no device mismatch)
        sigma = self._sigmas_cpu[timestep_id].to(device=self.device, dtype=latents.dtype)
        weight = self._weights_cpu[timestep_id].to(device=self.device, dtype=torch.float32)
        timestep_val = float(self._timesteps_cpu[timestep_id].item())

        target = noise - latents
        noisy_latents = (1 - sigma) * latents + sigma * noise

        # 4) text embeddings (frozen)
        with torch.no_grad():
            context = _encode_prompt(self.text_encoder, input_ids, attention_mask)

        # 5) DiT forward (official interface uses per-sample lists)
        x_list = [noisy_latents[i] for i in range(noisy_latents.shape[0])]
        context_list = [context[i] for i in range(context.shape[0])]

        d_f, d_h, d_w = getattr(self.model_args, "dit_patch_size", [1, 2, 2])
        max_seq_len = 0
        for x in x_list:
            seq_len = x.shape[1] * (x.shape[2] // d_h) * (x.shape[3] // d_w)
            max_seq_len = max(max_seq_len, int(seq_len))

        # separated-timestep support (match wan_new behavior when enabled)
        # Use scheduler "timestep" value (sigma * 1000), not the discrete index.
        t = sigma.new_tensor([timestep_val])
        if getattr(self.model_args, "seperated_timestep", False) and getattr(
            self.model_args, "fuse_vae_embedding_in_latents", False
        ):
            # Expand to [B, L] and set first-frame patches to 0.
            t_list = []
            for i, x in enumerate(x_list):
                f, h, w = x.shape[1], x.shape[2], x.shape[3]
                spatial_patches = (h // d_h) * (w // d_w)
                seq_len = f * spatial_patches
                t_seq = torch.full((seq_len,), timestep_val, device=self.device, dtype=torch.float32)
                t_seq[:spatial_patches] = 0.0
                # pad to max_seq_len
                if seq_len < max_seq_len:
                    t_seq = torch.cat([t_seq, torch.zeros(max_seq_len - seq_len, device=self.device, dtype=torch.float32)])
                t_list.append(t_seq)
            t = torch.stack(t_list)  # [B, max_seq_len]
        else:
            # Shape [B]
            t = torch.full((noisy_latents.shape[0],), timestep_val, device=self.device, dtype=torch.float32)

        with self.maybe_enable_amp:
            noise_pred_list = model(
                x_list=x_list,
                context_list=context_list,
                t=t,
                seq_len=max_seq_len,
                y_list=None,
            )
            noise_pred = torch.stack(noise_pred_list)

            pred = (noise_pred, target, weight)
            loss = self.loss_fn(pred, labels)

        del noise_pred, noise_pred_list, pred
        loss.backward()
        return loss


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    trainer: Optional[WanNew2Trainer] = None
    try:
        trainer = WanNew2Trainer(config)
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed.")