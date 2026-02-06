"""Trainer registrations."""

# from .base import BaseNativeTrainer, create_lr_scheduler
from .fsdp import build_fsdp_trainer
from .fsdp2 import build_fsdp2_trainer
from .hf import build_hf_trainer
