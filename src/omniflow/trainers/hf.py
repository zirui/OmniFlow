"""HF trainer implementation and registration."""

import os
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import save_file as safe_save_file
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback, TrainingArguments

from omniflow.optim.adamw_fp32_state import AdamWFP32State
from omniflow.registry import register_trainer
from omniflow.schedulers.flow_match import FlowMatchScheduler
from omniflow.utils.train_utils import get_memory, resolve_dtype, set_seed


class WanVideoCallback(TrainerCallback):
    """Callback to freeze non-trainable modules at training start."""

    def on_train_begin(self, args, state, control, model=None, logs=None, **kwargs):
        model.freeze_except()
        logger.info(f"Trainable_modules: {model.trainable_modules}. Freezing other modules.")
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            allocated, reserved, max_alloc = get_memory()
            logger.info(
                f"step={state.global_step}, "
                f"loss={logs['loss']:.4f}, "
                f"allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max_alloc={max_alloc:.2f}GB"
            )


class WanVideoTrainer(HFTrainer):
    """Custom trainer for WanVideo diffusion training."""

    def __init__(self, *args, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(WanVideoCallback())
        kwargs["callbacks"] = callbacks

        super().__init__(*args, **kwargs)

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        logger.info(f"Setting timesteps for diffusion training: {len(self.scheduler.timesteps)} steps")

        if os.environ.get("FIXED_SEED"):
            try:
                seed = int(os.environ["FIXED_SEED"])
                set_seed(seed)
                logger.info(f"Global seed set to {seed}")
            except ValueError:
                raise ValueError("FIXED_SEED must be an integer")

    def create_optimizer(self):
        super().create_optimizer()
        if (
            getattr(self.args, "bf16", False)
            and isinstance(self.model, FSDP)
            and os.getenv("FP32_MASTER_WEIGHTS", "0") == "1"
        ):
            logger.info("Using fp32-state AdamW to match DeepSpeed bf16 behavior.")
            self.optimizer = AdamWFP32State(
                self.optimizer.param_groups,
                lr=float(self.args.learning_rate),
                betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
                eps=float(self.args.adam_epsilon),
            )

        logger.info(
            f"Optimizer groups: Decay={len(self.optimizer.param_groups[0]['params'])}, No-Decay={len(self.optimizer.param_groups[1]['params'])}"
        )
        return self.optimizer

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Any,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        # Optional batch preparation hook (model-agnostic).
        # When using a raw collator, HF Trainer will receive `inputs` as a list of
        # dicts. If a `processing_class.prepare_batch` hook is provided, use it
        # to convert raw inputs into the dict expected by `model(batch, scheduler)`.
        processing_class = getattr(self, "processing_class", None)
        prepare_batch = getattr(processing_class, "prepare_batch", None)
        if callable(prepare_batch):
            inputs = prepare_batch(
                batch=inputs,
                device=model.device if hasattr(model, "device") else self.args.device,
                dtype=resolve_dtype(self.args),
            )

        if not isinstance(inputs, dict):
            raise TypeError(
                "HF trainer expected `inputs` as dict after prepare_batch. "
                f"Got: {type(inputs)}. If you use RawBatchCollator, implement processor.prepare_batch()."
            )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device if hasattr(model, "device") else self.args.device)

        outputs = model(inputs, self.scheduler)
        return outputs["loss"]

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not self.is_world_process_zero():
            return

        model = self.model
        core_model = model.module if hasattr(model, "module") else model

        if not hasattr(core_model, "dit"):
            logger.warning("save_model: model has no `dit` attribute; skipping save.")
            return

        if isinstance(model, FSDP):
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_state = model.state_dict()
            dit_state_dict = {k[len("dit.") :]: v for k, v in full_state.items() if k.startswith("dit.")}
        else:
            dit_state_dict = core_model.dit.state_dict()

        if not dit_state_dict:
            logger.warning("save_model: DiT state dict is empty; skipping save.")
            return

        save_path = os.path.join(output_dir, "dit_model.safetensors")
        safe_save_file(dit_state_dict, save_path)
        logger.info(f"Saved DiT weights to {save_path}")

        if hasattr(core_model, "config"):
            try:
                core_model.config.save_pretrained(output_dir)
            except Exception as exc:
                logger.warning(f"save_model: failed to save config: {exc}")


@register_trainer("hf")
def build_hf_trainer(*, model, dataset, processor, trainer_args: dict):
    training_args = TrainingArguments(**trainer_args)
    logger.info(f"Training arguments: {training_args}")

    fsdp_enabled = bool(getattr(training_args, "fsdp", None))
    if getattr(training_args, "bf16", False) and fsdp_enabled:
        if hasattr(model, "dit") and model.dit is not None:
            model.dit.to(dtype=torch.bfloat16)
        if hasattr(model, "vae") and model.vae is not None:
            model.vae.to(dtype=torch.bfloat16)
        if hasattr(model, "text_encoder") and model.text_encoder is not None:
            model.text_encoder.to(dtype=torch.bfloat16)
        logger.info("FSDP+bf16: casted `dit`, `vae`, and `text_encoder` to bf16")

    output_dir = getattr(training_args, "output_dir", None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    trainer = WanVideoTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.get_collator(),
        processing_class=processor,
    )
    return trainer
