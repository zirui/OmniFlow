from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torchtitan.protocols.model import ModelProtocol

from omniflow.models.wan_new2.wan_dit import WanModel as WanDiT
from .wan_args import WanNew2ModelArgs


class WanNew2DiTModel(nn.Module, ModelProtocol):
    """
    TorchTitan model part for wan_new2.

    Important:
    - This is *only* the DiT backbone.
    - VAE + T5 preprocessing is handled in `wan_torchtitan_new2/train.py` (Trainer subclass),
      similar to how Flux does encoders outside the model.
    """

    def __init__(self, model_args: WanNew2ModelArgs):
        super().__init__()
        self.model_args = model_args

        self.dit = WanDiT(
            model_type=model_args.model_type,
            patch_size=tuple(model_args.dit_patch_size),
            text_len=int(getattr(model_args, "text_len", 512)),
            in_dim=int(model_args.dit_in_channels),
            dim=int(model_args.dit_hidden_size),
            ffn_dim=int(model_args.dit_intermediate_size),
            freq_dim=int(model_args.dit_freq_dim),
            text_dim=int(model_args.dit_text_dim),
            out_dim=int(model_args.dit_out_channels),
            num_heads=int(model_args.dit_num_heads),
            num_layers=int(model_args.dit_num_layers),
            window_size=tuple(model_args.dit_window_size),
            qk_norm=bool(model_args.dit_qk_norm),
            cross_attn_norm=bool(model_args.dit_cross_attn_norm),
            eps=float(model_args.dit_eps),
        )

    def init_weights(self, buffer_device=None):
        """
        TorchTitan builds models under `torch.device("meta")`, then materializes
        parameters via `to_empty()` + `init_weights()`.

        The vendored official Wan DiT keeps RoPE freqs as a plain Tensor attribute
        (not a parameter/buffer). When constructed on meta, `self.dit.freqs` stays
        as a meta tensor and will crash on first forward (`Cannot copy out of meta tensor`).

        We rebuild `dit.freqs` here on a real device (CPU) so it can be moved to CUDA
        lazily in forward.
        """
        try:
            freqs = getattr(self.dit, "freqs", None)
            if isinstance(freqs, torch.Tensor) and getattr(freqs, "is_meta", False):
                from omniflow.models.wan_new2.wan_dit import rope_params

                dim = int(getattr(self.dit, "dim"))
                num_heads = int(getattr(self.dit, "num_heads"))
                assert (dim % num_heads) == 0
                d = dim // num_heads
                rebuilt = torch.cat(
                    [
                        rope_params(1024, d - 4 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                    ],
                    dim=1,
                )
                self.dit.freqs = rebuilt
        except Exception:
            # Best-effort; if anything goes wrong, let the real error surface in forward.
            pass
        return

    def forward(
        self,
        *,
        x_list: list[torch.Tensor],
        context_list: list[torch.Tensor],
        t: torch.Tensor,
        seq_len: int,
        y_list: Optional[list[torch.Tensor]] = None,
    ):
        # WanDiT API: x/context as per-sample lists.
        return self.dit(x=x_list, t=t, context=context_list, seq_len=seq_len, y=y_list)


def maybe_enable_dit_gradient_checkpointing(model: nn.Module, enabled: bool) -> None:
    """
    WanNew2 DiT has an internal checkpoint path controlled by `dit.gradient_checkpointing`.
    We keep this off by default and prefer TorchTitan activation checkpoint wrappers.
    """
    try:
        if hasattr(model, "dit") and hasattr(model.dit, "gradient_checkpointing"):
            model.dit.gradient_checkpointing = bool(enabled)
    except Exception:
        pass

