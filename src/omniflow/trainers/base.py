"""
Base trainer with shared logic for native PyTorch trainers (FSDP, FSDP2).

This module eliminates code duplication between fsdp.py and fsdp2.py by
extracting common functionality: config parsing, optimizer creation,
LR scheduling, training loop, logging, and W&B integration.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager

import torch
from loguru import logger

from omniflow.optim.adamw_fp32_state import AdamWFP32State
from omniflow.schedulers.flow_match import FlowMatchScheduler
from omniflow.utils.train_utils import get_memory, resolve_dtype, set_seed

try:
    import wandb
except ImportError:
    wandb = None


def create_lr_scheduler(optimizer, scheduler_type, warmup_steps, total_steps):
    """
    Create LR scheduler with warmup support.

    Supports: constant, constant_with_warmup, linear, cosine, polynomial.
    Shared between FSDP and FSDP2 trainers for consistency.
    """
    if total_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    def linear_warmup(step):
        if warmup_steps == 0:
            return 1.0
        return min(1.0, float(step) / float(max(1, warmup_steps)))

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
    lambdas = {
        "constant": lambda step: 1.0,
        "constant_with_warmup": constant_with_warmup,
        "linear": linear_decay,
        "cosine": cosine_decay,
        "cosine_with_restarts": cosine_decay,
        "polynomial": polynomial_decay,
    }
    lr_lambda = lambdas.get(scheduler_type)
    if lr_lambda is None:
        logger.warning(f"Unknown lr_scheduler_type={scheduler_type}, falling back to constant.")
        def lr_lambda(step):
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class BaseNativeTrainer:
    """
    Shared base class for native PyTorch trainers (FSDP, FSDP2).

    Subclasses must implement:
      - _apply_parallelism(): set up distributed wrapping / sharding
      - save_model(): save final model

    Subclasses may override:
      - _grad_sync_context(is_update_step): gradient sync during accumulation
      - _clip_grad_norm(): gradient clipping strategy
      - _save_checkpoint(): periodic checkpoint saving
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
        train_dataset,
        data_collator,
        processing_class,
        rank: int,
        world_size: int,
        local_rank: int,
    ):
        self.model = model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")

        # --- Config extraction ---
        self.output_dir = self.args.get("output_dir", "./output")
        self.logging_steps = int(self.args.get("logging_steps", 1))
        self.save_steps = int(self.args.get("save_steps", 0))
        self.max_steps = int(self.args.get("max_steps", -1) if self.args.get("max_steps") is not None else -1)
        self.grad_accum_steps = int(self.args.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(self.args.get("max_grad_norm", 1.0))
        self.num_train_epochs = int(self.args.get("num_train_epochs", 1))

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        # --- Seeding ---
        seed = self.args.get("seed")
        if seed is not None:
            set_seed(int(seed))
        if os.environ.get("FIXED_SEED"):
            set_seed(int(os.environ["FIXED_SEED"]))

        # --- Gradient Checkpointing ---
        if self.args.get("gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "dit") and hasattr(self.model.dit, "gradient_checkpointing"):
                self.model.dit.gradient_checkpointing = True
            if self.rank == 0:
                logger.info("Gradient checkpointing enabled")

        # --- W&B ---
        self._setup_wandb()

        # --- Parallelism (subclass hook) ---
        # Subclass sets self.sp_group (Ulysses SP group) if SP is enabled.
        self.sp_group = None
        self._apply_parallelism()

        # --- DataLoader ---
        self.train_dataset = train_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator

        # When SP is enabled, all ranks in the same SP group process the same sample.
        # DistributedSampler should use DP-only rank/size so SP peers get identical data.
        dp_world_size = world_size
        dp_rank = rank
        if self.sp_group is not None:
            import torch.distributed as dist
            sp_size = dist.get_world_size(self.sp_group)
            dp_world_size = world_size // sp_size
            dp_rank = rank // sp_size

        self.sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=self.args.get("shuffle", True),
        )

        num_workers = int(self.args.get("dataloader_num_workers", 4) or 0)
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.get("per_device_train_batch_size", 1),
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # --- Optimizer ---
        self.optimizer = self._create_optimizer()

        # --- LR Scheduler ---
        steps_per_epoch = math.ceil(len(self.dataloader) / max(1, self.grad_accum_steps))
        self.total_steps = self.max_steps if self.max_steps > 0 else self.num_train_epochs * steps_per_epoch
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            self.args.get("lr_scheduler_type", "constant"),
            int(self.args.get("warmup_steps", 0)),
            self.total_steps,
        )

        # --- Diffusion Scheduler (configurable from YAML) ---
        scheduler_cfg = self.args.get("flow_match_scheduler", {}) or {}
        self.scheduler = FlowMatchScheduler(
            shift=float(scheduler_cfg.get("shift", 5)),
            sigma_min=float(scheduler_cfg.get("sigma_min", 0.0)),
            extra_one_step=bool(scheduler_cfg.get("extra_one_step", True)),
        )
        self.scheduler.set_timesteps(
            int(scheduler_cfg.get("num_train_timesteps", 1000)),
            training=True,
        )

        self.global_step = 0

    # ------------------------------------------------------------------ #
    #                       Subclass hooks                                 #
    # ------------------------------------------------------------------ #

    def _apply_parallelism(self):
        """Set up distributed parallelism. Called during __init__."""
        raise NotImplementedError

    @contextmanager
    def _grad_sync_context(self, is_update_step: bool):
        """Context manager for gradient sync control during accumulation."""
        yield

    def _clip_grad_norm(self) -> float:
        """Clip gradient norm. Returns the total norm value."""
        if self.max_grad_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            return norm.item() if isinstance(norm, torch.Tensor) else float(norm)
        return 0.0

    def _save_checkpoint(self):
        """Save checkpoint at save_steps intervals. Override for custom strategies."""
        pass

    # ------------------------------------------------------------------ #
    #                       Common methods                                 #
    # ------------------------------------------------------------------ #

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

    def _resolve_dtype(self) -> torch.dtype:
        return resolve_dtype(self.args)

    def _create_optimizer(self):
        lr = float(self.args.get("learning_rate", 1e-4))
        wd = float(self.args.get("weight_decay", 0.01))
        betas = (
            float(self.args.get("adam_beta1", 0.9)),
            float(self.args.get("adam_beta2", 0.999)),
        )
        eps = float(self.args.get("adam_epsilon", 1e-8))

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_kwargs = {"params": params, "lr": lr, "betas": betas, "eps": eps, "weight_decay": wd}

        optimizer = None
        try:
            optimizer = torch.optim.AdamW(**optimizer_kwargs, fused=True)
            if self.rank == 0:
                logger.info("Optimizer: torch.optim.AdamW(fused=True)")
        except TypeError:
            try:
                optimizer = torch.optim.AdamW(**optimizer_kwargs, foreach=True)
                if self.rank == 0:
                    logger.info("Optimizer: torch.optim.AdamW(foreach=True)")
            except TypeError:
                optimizer = torch.optim.AdamW(**optimizer_kwargs)
                if self.rank == 0:
                    logger.info("Optimizer: torch.optim.AdamW(default)")

        if (self.args.get("bf16", False) or self.args.get("fp16", False)) and os.getenv(
            "FP32_MASTER_WEIGHTS", "0"
        ) == "1":
            if self.rank == 0:
                logger.info("FP32_MASTER_WEIGHTS=1: using AdamWFP32State (fp32 master weights + fp32 moments).")
            optimizer = AdamWFP32State(
                optimizer.param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=wd,
            )

        return optimizer

    def compute_loss(self, batch):
        """Prepare batch and compute training loss."""
        prepare_batch = getattr(self.processing_class, "prepare_batch", None)
        if callable(prepare_batch):
            batch = prepare_batch(
                batch=batch,
                device=self.device,
                dtype=self._resolve_dtype(),
            )
        # Ensure all tensors are on the correct device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        # Pass SP group so model can shard sequences across SP ranks
        if self.sp_group is not None:
            batch["sp_group"] = self.sp_group

        # Use explicit training entry point if available (GenAIModel interface)
        forward_train = getattr(self.model, "forward_train", None)
        if callable(forward_train):
            outputs = forward_train(batch, scheduler=self.scheduler)
        else:
            outputs = self.model(batch, self.scheduler)
        return outputs["loss"]

    def _log_step(
        self,
        loss_value: float,
        grad_norm: float = 0.0,
        step_time: float | None = None,
        elapsed: float | None = None,
        eta_seconds: float | None = None,
    ):
        """Log training metrics. Format matches test regex expectations."""
        if self.rank != 0:
            return
        alloc, res, max_mem = get_memory()
        lr = self.optimizer.param_groups[0]["lr"]

        # NOTE: The "step=... loss=... mem=.../...GB" format is parsed by
        # tests/test_wan_fsdp2_training_baseline.py. Do not change it.
        msg = f"step={self.global_step} loss={loss_value:.4f} mem={alloc:.2f}/{res:.2f}GB gnorm={grad_norm:.4f}"
        if step_time is not None:
            msg += f" step_time={step_time:.2f}s"
        if elapsed is not None:
            msg += f" elapsed={elapsed / 60:.2f}m"
        if eta_seconds is not None:
            msg += f" eta={eta_seconds / 60:.2f}m"
        logger.info(msg)

        if self.use_wandb:
            payload = {
                "train/loss": loss_value,
                "train/step": self.global_step,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "mem/allocated_gb": alloc,
                "mem/reserved_gb": res,
                "mem/max_alloc_gb": max_mem,
            }
            if step_time is not None:
                payload["time/step_s"] = step_time
            if elapsed is not None:
                payload["time/elapsed_s"] = elapsed
            if eta_seconds is not None:
                payload["time/eta_s"] = eta_seconds
            wandb.log(payload, step=self.global_step)

    def train(self):
        if self.rank == 0:
            logger.info("Starting training...")

        # Ensure frozen state (idempotent)
        core = getattr(self.model, "module", self.model)
        if hasattr(core, "freeze_except"):
            core.freeze_except()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        last_log_time = start_time

        for epoch in range(self.num_train_epochs):
            self.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.dataloader):
                is_update_step = ((batch_idx + 1) % max(1, self.grad_accum_steps)) == 0

                with self._grad_sync_context(is_update_step):
                    loss = self.compute_loss(batch)
                    loss = loss / max(1, self.grad_accum_steps)
                    loss.backward()

                if is_update_step:
                    grad_norm = self._clip_grad_norm()

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        loss_val = loss.detach().float().item() * max(1, self.grad_accum_steps)
                        now = time.time()
                        step_time = now - last_log_time
                        last_log_time = now
                        elapsed = now - start_time
                        steps_left = max(0, self.total_steps - self.global_step)
                        eta_seconds = step_time * steps_left
                        self._log_step(
                            loss_val,
                            grad_norm=grad_norm,
                            step_time=step_time,
                            elapsed=elapsed,
                            eta_seconds=eta_seconds,
                        )

                    # Periodic save
                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint()

                    # Early termination
                    if self.max_steps > 0 and self.global_step >= self.max_steps:
                        return

            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        if self.rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Training finished in {elapsed / 60:.2f} min")

    def save_model(self):
        """Save final model. Override in subclass."""
        raise NotImplementedError
