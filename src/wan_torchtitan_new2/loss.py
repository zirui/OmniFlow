from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import LossFunction
from torchtitan.distributed import ParallelDims


class WanNew2Loss(nn.Module):
    """
    Loss function for wan_new2 training.

    `pred` is expected to be a tuple:
      (noise_pred, target, weight)
    where `weight` is a scalar tensor already on the correct device.
    """

    def forward(self, pred, labels):  # labels is unused (kept for Trainer signature)
        noise_pred, target, weight = pred
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        return loss * weight


def build_wan_new2_loss(job_config, parallel_dims: ParallelDims, ft_manager) -> LossFunction:
    return WanNew2Loss()

