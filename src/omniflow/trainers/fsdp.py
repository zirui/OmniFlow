"""Native torch FSDP trainer"""

from __future__ import annotations

import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from loguru import logger
from safetensors.torch import save_file as safe_save_file
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from importlib import import_module
import re
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler

from omniflow.optim.adamw_fp32_state import AdamWFP32State
from omniflow.registry import register_trainer
from omniflow.schedulers.flow_match import FlowMatchScheduler
from omniflow.utils.train_utils import get_memory, set_seed, resolve_dtype

try:
    import wandb
except Exception:
    wandb = None


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(0)
    return rank, world_size, local_rank


def _resolve_dtype(trainer_args: dict):
    # Backward-compatible shim; prefer `omniflow.utils.train_utils.resolve_dtype`.
    return resolve_dtype(trainer_args)


def _create_lr_scheduler(optimizer, scheduler_type, warmup_steps, total_steps):
    if total_steps <= 0:
        return None

    def linear_warmup(step):
        if warmup_steps == 0:
            return 1.0
        return min(1.0, float(step) / float(warmup_steps))

    def linear_decay(step):
        if step <= warmup_steps:
            return linear_warmup(step)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    def cosine_decay(step):
        if step <= warmup_steps:
            return linear_warmup(step)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def constant_with_warmup(step):
        return linear_warmup(step)

    def polynomial_decay(step, power=1.0):
        if step <= warmup_steps:
            return linear_warmup(step)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, (1.0 - progress) ** power)

    scheduler_type = (scheduler_type or "constant").lower()
    if scheduler_type == "constant":
        lr_lambda = lambda step: 1.0
    elif scheduler_type == "constant_with_warmup":
        lr_lambda = constant_with_warmup
    elif scheduler_type == "linear":
        lr_lambda = linear_decay
    elif scheduler_type == "cosine":
        lr_lambda = cosine_decay
    elif scheduler_type == "cosine_with_restarts":
        lr_lambda = cosine_decay
    elif scheduler_type == "polynomial":
        lr_lambda = polynomial_decay
    else:
        logger.warning(f"Unknown lr_scheduler_type={scheduler_type}, falling back to constant.")
        lr_lambda = lambda step: 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class FSDPTrainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        data_collator,
        processing_class,
        rank,
        world_size,
        local_rank,
    ):
        self.model = model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")

        self.output_dir = self.args.get("output_dir", "./output")
        self.logging_steps = int(self.args.get("logging_steps", 1))
        self.save_steps = int(self.args.get("save_steps", 0))
        self.max_steps = int(self.args.get("max_steps", -1)) if self.args.get("max_steps") is not None else -1
        self.grad_accum_steps = int(self.args.get("gradient_accumulation_steps", 1))
        # Match HF TrainingArguments default (0.0) when not explicitly set.
        self.max_grad_norm = float(self.args.get("max_grad_norm", 1.0))
        self.num_train_epochs = int(self.args.get("num_train_epochs", 1))

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        if self.args.get("seed") is not None:
            set_seed(int(self.args["seed"]))

        if os.environ.get("FIXED_SEED"):
            try:
                set_seed(int(os.environ["FIXED_SEED"]))
                logger.info(f"Global seed set to {os.environ['FIXED_SEED']}")
            except ValueError:
                raise ValueError("FIXED_SEED must be an integer")

        if self.args.get("gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "dit") and hasattr(self.model.dit, "gradient_checkpointing"):
                self.model.dit.gradient_checkpointing = True
            logger.info("Gradient checkpointing enabled")

        if self.args.get("bf16", False) and hasattr(self.model, "dit") and self.model.dit is not None:
            self.model.dit.to(dtype=torch.bfloat16)
            logger.info("FSDP+bf16: casted `dit` to bf16")

        self._setup_wandb()

        self.model.to(self.device)
        self._wrap_fsdp()

        self.train_dataset = train_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator
        self.sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=self.args.get("shuffle", True),
        )

        num_workers = int(self.args.get("dataloader_num_workers", 4) or 0)

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.get("per_device_train_batch_size", 1),
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True,
            worker_init_fn=None,
            persistent_workers=bool(
                self.args.get(
                    "dataloader_persistent_workers",
                    (self.args.get("dataloader_num_workers", 0) or 0) > 0,
                )
            ),
            prefetch_factor=self.args.get("dataloader_prefetch_factor", 2)
            if (self.args.get("dataloader_num_workers", 0) or 0) > 0
            else None,
        )

        self.optimizer = self._create_optimizer()

        steps_per_epoch = math.ceil(len(self.dataloader) / max(1, self.grad_accum_steps))
        total_steps = self.max_steps if self.max_steps and self.max_steps > 0 else self.num_train_epochs * steps_per_epoch
        self.lr_scheduler = _create_lr_scheduler(
            self.optimizer,
            self.args.get("lr_scheduler_type", "constant"),
            int(self.args.get("warmup_steps", 0)),
            total_steps,
        )

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.global_step = 0
        self.model.train()
        torch.cuda.reset_peak_memory_stats()

    def _setup_wandb(self):
        self.use_wandb = False
        if self.rank != 0:
            return
        if str(self.args.get("report_to", "")).lower() != "wandb":
            return
        if self.args.get("use_wandb") is False:
            return
        if wandb is None:
            logger.warning("W&B requested but wandb is not installed.")
            return

        project = self.args.get("wandb_project") or os.environ.get("WANDB_PROJECT", "omniflow")
        run_name = self.args.get("wandb_name") or self.args.get("run_name")
        wandb_dir = self.args.get("wandb_dir") or os.environ.get("WANDB_DIR")
        wandb.init(project=project, name=run_name, dir=wandb_dir, config=self.args)
        self.use_wandb = True

    def _wrap_fsdp(self):
        # Avoid FSDP wrapping for single GPU runs.
        if int(getattr(self, "world_size", 1)) == 1 and self.args.get("fsdp_disable_on_single_gpu", True):
            if self.rank == 0:
                logger.info("world_size=1: skipping FSDP wrapping for speed (set fsdp_disable_on_single_gpu=false to override)")
            core_model = self.model
            if hasattr(core_model, "freeze_except"):
                core_model.freeze_except()
            return

        fsdp_config = self.args.get("fsdp_config", {}) or {}
        sharding_strategy_str = self.args.get("fsdp_sharding_strategy", "FULL_SHARD")
        sharding_strategy = getattr(ShardingStrategy, sharding_strategy_str, ShardingStrategy.FULL_SHARD)

        mixed_precision_dtype = _resolve_dtype(self.args)
        mp_policy = None
        if mixed_precision_dtype != torch.float32:
            # Match HF/Accelerate default: keep reductions/buffers in bf16 for bf16 runs.
            mp_policy = MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=mixed_precision_dtype,
            )

        backward_prefetch = None
        backward_prefetch_str = fsdp_config.get("backward_prefetch")
        if isinstance(backward_prefetch_str, str):
            backward_prefetch_str = backward_prefetch_str.lower()
            if backward_prefetch_str == "backward_pre":
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
            elif backward_prefetch_str == "backward_post":
                backward_prefetch = BackwardPrefetch.BACKWARD_POST

        ignored_modules = []
        ignored_spec = fsdp_config.get("ignored_modules")
        if ignored_spec:
            try:
                if isinstance(ignored_spec, (list, tuple, set)):
                    ignore_names = {str(item).strip() for item in ignored_spec if str(item).strip()}
                    for name, module in self.model.named_modules():
                        if name in ignore_names:
                            ignored_modules.append(module)
                else:
                    regex = re.compile(str(ignored_spec))
                    for name, module in self.model.named_modules():
                        if regex.fullmatch(name):
                            ignored_modules.append(module)
            except Exception as exc:
                logger.warning(f"Invalid ignored_modules spec: {exc}")

        auto_wrap_policy = None
        layer_cls = self.args.get("fsdp_transformer_layer_cls_to_wrap")
        if layer_cls:
            try:
                if isinstance(layer_cls, (list, tuple)):
                    class_paths = [str(item).strip() for item in layer_cls if str(item).strip()]
                else:
                    class_paths = [name.strip() for name in str(layer_cls).split(",") if name.strip()]
                cls_set = set()
                for class_path in class_paths:
                    if "." not in class_path:
                        # Match HF behavior: resolve by class name in the model graph.
                        resolved = False
                        for _, module in self.model.named_modules():
                            if module.__class__.__name__ == class_path:
                                cls_set.add(module.__class__)
                                resolved = True
                        if not resolved:
                            logger.warning(
                                f"fsdp_transformer_layer_cls_to_wrap expects fully qualified paths; "
                                f"could not resolve '{class_path}' from model modules"
                            )
                        continue
                    module_path, cls_name = class_path.rsplit(".", 1)
                    module = import_module(module_path)
                    cls_set.add(getattr(module, cls_name))
                if cls_set:
                    auto_wrap_policy = partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=cls_set,
                    )
            except Exception as exc:
                logger.warning(f"Failed to set auto_wrap_policy: {exc}")

        self.model = FSDP(
            self.model,
            device_id=self.device,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            use_orig_params=fsdp_config.get("use_orig_params", True),
            forward_prefetch=fsdp_config.get("forward_prefetch", True),
            backward_prefetch=backward_prefetch,
            sync_module_states=fsdp_config.get("sync_module_states", False),
            limit_all_gathers=fsdp_config.get("limit_all_gathers", False),
            auto_wrap_policy=auto_wrap_policy,
            ignored_modules=ignored_modules if ignored_modules else None,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.get("cpu_offload", False) else None,
        )
        # Match HF callback behavior: ensure only trainable modules are in train mode.
        core_model = self.model.module if isinstance(self.model, FSDP) else self.model
        if hasattr(core_model, "freeze_except"):
            core_model.freeze_except()

    def _create_optimizer(self):
        learning_rate = float(self.args.get("learning_rate", 5e-5))
        weight_decay = float(self.args.get("weight_decay", 0.01))
        betas = (
            float(self.args.get("adam_beta1", 0.9)),
            float(self.args.get("adam_beta2", 0.999)),
        )
        eps = float(self.args.get("adam_epsilon", 1e-8))
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer_kwargs = dict(
            params=params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        optimizer = None
        try:
            optimizer = torch.optim.AdamW(**optimizer_kwargs, fused=True)
            if self.rank == 0:
                logger.info("Optimizer: torch.optim.AdamW(fused=True)")
        except TypeError:
            # Older torch or CPU path: fall back to foreach (usually faster than per-tensor)
            try:
                optimizer = torch.optim.AdamW(**optimizer_kwargs, foreach=True)
                if self.rank == 0:
                    logger.info("Optimizer: torch.optim.AdamW(foreach=True)")
            except TypeError:
                optimizer = torch.optim.AdamW(**optimizer_kwargs)
                if self.rank == 0:
                    logger.info("Optimizer: torch.optim.AdamW(default)")

        if self.args.get("bf16", False) and os.getenv("FP32_MASTER_WEIGHTS", "0") == "1":
            logger.info("FP32_MASTER_WEIGHTS=1: using AdamWFP32State (fp32 master weights + fp32 moments).")
            optimizer = AdamWFP32State(
                optimizer.param_groups,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        
        # DEBUG: Log optimizer param dtypes
        for i, group in enumerate(optimizer.param_groups):
            first_param = group["params"][0]
            logger.info(f"Optimizer Group {i} first param dtype: {first_param.dtype}")

        return optimizer

    def _log_step(self, loss_value, step_time=None, elapsed=None, eta_seconds=None):
        if self.rank != 0:
            return
        allocated, reserved, max_alloc = get_memory()
        lr = self.optimizer.param_groups[0]["lr"]
        log_message = (
            f"step={self.global_step}, "
            f"loss={loss_value:.4f}, "
            f"lr={lr:.6g}, "
            f"allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, "
            f"max_alloc={max_alloc:.2f}GB"
        )
        if step_time is not None:
            log_message += f", step_time={step_time:.2f}s"
        if elapsed is not None:
            log_message += f", elapsed={elapsed/60:.2f}m"
        if eta_seconds is not None:
            log_message += f", eta={eta_seconds/60:.2f}m"
        logger.info(log_message)
        if self.use_wandb:
            payload = {
                "train/loss": loss_value,
                "train/lr": lr,
                "mem/allocated_gb": allocated,
                "mem/reserved_gb": reserved,
                "mem/max_alloc_gb": max_alloc,
                "train/global_step": self.global_step,
            }
            if step_time is not None:
                payload["time/step_s"] = step_time
            if elapsed is not None:
                payload["time/elapsed_s"] = elapsed
            if eta_seconds is not None:
                payload["time/eta_s"] = eta_seconds
            wandb.log(payload, step=self.global_step)

    def _save_dit(self, save_path):
        core_model = self.model.module if isinstance(self.model, FSDP) else self.model
        if not hasattr(core_model, "dit"):
            logger.warning("save_model: model has no `dit` attribute; skipping save.")
            return

        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_state = self.model.state_dict()
            dit_state_dict = {k[len("dit.") :]: v for k, v in full_state.items() if k.startswith("dit.")}
        else:
            dit_state_dict = core_model.dit.state_dict()

        if not dit_state_dict:
            logger.warning("save_model: DiT state dict is empty; skipping save.")
            return

        safe_save_file(dit_state_dict, save_path)
        logger.info(f"Saved DiT weights to {save_path}")

        if hasattr(core_model, "config"):
            try:
                core_model.config.save_pretrained(self.output_dir)
            except Exception as exc:
                logger.warning(f"save_model: failed to save config: {exc}")

    def compute_loss(self, batch):
        # Optional batch preparation hook (model-agnostic).
        # If provided, `processing_class.prepare_batch` should convert the raw
        # dataloader output into the dict expected by `model(batch, scheduler)`.
        prepare_batch = getattr(self.processing_class, "prepare_batch", None)
        if callable(prepare_batch):
            batch = prepare_batch(
                batch=batch,
                device=self.device,
                dtype=_resolve_dtype(self.args),
            )
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        outputs = self.model(batch, self.scheduler)
        return outputs["loss"]

    def train(self):
        if self.rank == 0:
            logger.info("Starting training loop...")
        # Match HF callback: freeze non-trainable modules at train start
        core_model = self.model.module if isinstance(self.model, FSDP) else self.model
        if hasattr(core_model, "freeze_except"):
            core_model.freeze_except()
        self.optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        last_log_time = start_time
        for epoch in range(self.num_train_epochs):
            self.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.dataloader):
                is_update_step = (batch_idx + 1) % self.grad_accum_steps == 0
                if isinstance(self.model, FSDP) and not is_update_step:
                    with self.model.no_sync():
                        loss = self.compute_loss(batch)
                        loss = loss / max(1, self.grad_accum_steps)
                        loss.backward()
                else:
                    loss = self.compute_loss(batch)
                    loss = loss / max(1, self.grad_accum_steps)
                    loss.backward()

                if is_update_step:
                    if self.max_grad_norm and self.max_grad_norm > 0:
                        if isinstance(self.model, FSDP):
                            self.model.clip_grad_norm_(self.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    loss_value = loss.detach().float().item() * max(1, self.grad_accum_steps)

                    if self.global_step % self.logging_steps == 0:
                        now = time.time()
                        step_time = now - last_log_time
                        last_log_time = now
                        elapsed = now - start_time
                        total_steps = (
                            self.max_steps
                            if self.max_steps and self.max_steps > 0
                            else self.num_train_epochs
                            * math.ceil(len(self.dataloader) / max(1, self.grad_accum_steps))
                        )
                        steps_left = max(0, total_steps - self.global_step)
                        eta_seconds = step_time * steps_left
                        self._log_step(
                            loss_value,
                            step_time=step_time,
                            elapsed=elapsed,
                            eta_seconds=eta_seconds,
                        )

                    if self.save_steps and self.global_step % self.save_steps == 0 and self.rank == 0:
                        save_path = os.path.join(self.output_dir, f"dit_checkpoint-{self.global_step}.safetensors")
                        self._save_dit(save_path)

                    if self.max_steps > 0 and self.global_step >= self.max_steps:
                        break

            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        if self.rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Training finished in {elapsed/60:.2f} min")
            save_path = os.path.join(self.output_dir, "dit_model.safetensors")
            self._save_dit(save_path)

        if self.use_wandb and self.rank == 0:
            wandb.finish()

    def save_model(self):
        if self.rank == 0:
            save_path = os.path.join(self.output_dir, "dit_model.safetensors")
            self._save_dit(save_path)
        if dist.is_initialized():
            dist.destroy_process_group()


@register_trainer("fsdp")
def build_fsdp_trainer(*, model, dataset, processor, trainer_args: dict):
    rank, world_size, local_rank = setup_distributed()
    if rank == 0:
        logger.info("Building native FSDP trainer")

    return FSDPTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        data_collator=dataset.get_collator(),
        processing_class=processor,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
