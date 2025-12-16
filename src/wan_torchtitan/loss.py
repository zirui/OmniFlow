# WanLoss
from .model.wan_video_scheduler import FlowMatchScheduler
from torchtitan.components.loss import LossFunction
from torchtitan.distributed import ParallelDims

import torch.nn as nn
import torch.nn.functional as F


class WanLoss(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def forward(self, pred, labels):
        # pred is (noise_pred, training_target, timestep) from WanVideoModel.forward
        noise_pred, training_target, timestep = pred

        # Compute MSE loss
        loss = F.mse_loss(noise_pred.float(), training_target.float(), reduction="mean")
        loss = loss * self.scheduler.training_weight(timestep)

        return loss


def build_wan_loss(job_config, parallel_dims: ParallelDims, ft_manager) -> LossFunction:
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return WanLoss(scheduler)
