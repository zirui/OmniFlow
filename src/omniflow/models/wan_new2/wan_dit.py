# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Vendored from `Wan2.2/wan/modules/model.py` with minimal changes:
# - remove diffusers dependencies (ModelMixin/ConfigMixin/register_to_config)
# - keep pure PyTorch modeling as close as possible
# - provide a `DiTBlock` alias class so existing OmniFlow YAML can keep
#   `fsdp_transformer_layer_cls_to_wrap: "DiTBlock"`

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .attention_backend import attention

__all__ = ["WanModel", "DiTBlock"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    # IMPORTANT:
    # Returning float32 here creates an extra full-size copy of q/k (and thus
    # doubles memory) on long sequences, which can OOM even when the same token
    # setting works in `wan_new`. Keep dtype consistent with input activations.
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # Keep numerical stability by normalizing in fp32, but do NOT rely on
        # module parameters being fp32 (FSDP/bf16 can cast weights/bias).
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        y = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
        return y.type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], (F, H, W)
            freqs(Tensor): Rope freqs
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x_):
            # FSDP/bf16 training can keep module weights in bf16 while upstream
            # code uses some float32 math (e.g., rope_apply -> float32). PyTorch
            # Linear requires input/weight dtypes to match, so we explicitly cast.
            x_ = x_.to(dtype=self.q.weight.dtype)
            q = self.norm_q(self.q(x_)).view(b, s, n, d)
            k = self.norm_k(self.k(x_)).view(b, s, n, d)
            v = self.v(x_).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            dtype=self.q.weight.dtype,
        )

        # `rope_apply` returns float32 by design; cast back to module dtype
        # before output projection to avoid matmul dtype mismatch.
        x = x.to(dtype=self.o.weight.dtype)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        x = x.to(dtype=self.q.weight.dtype)
        context = context.to(dtype=self.k.weight.dtype)

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = attention(q, k, v, k_lens=context_lens, dtype=self.q.weight.dtype)

        x = x.to(dtype=self.o.weight.dtype)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        # Memory-critical:
        # `e` can be very large ([B, seq_len, 6, dim]). Keeping it in fp32 and
        # doing broadcast add in fp32 easily OOMs for Wan2.1 (more tokens).
        # Align to x.dtype (bf16/fp16) like `wan_new` implementation.
        e = e.to(dtype=x.dtype)
        modulation = self.modulation.to(dtype=x.dtype)
        e = (modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
        )
        x = x + y * e[2].squeeze(2)

        def cross_attn_ffn(x_, context_, context_lens_, e_):
            x_ = x_ + self.cross_attn(self.norm3(x_), context_, context_lens_)
            ffn_in = self.norm2(x_) * (1 + e_[4].squeeze(2)) + e_[3].squeeze(2)
            y_ = self.ffn(ffn_in)
            x_ = x_ + y_ * e_[5].squeeze(2)
            return x_

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


# OmniFlow YAML compatibility: FSDP auto-wrap often uses "DiTBlock".
class DiTBlock(WanAttentionBlock):
    pass


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        # Align modulation embedding dtype to x (saves memory vs fp32 broadcast).
        e = e.to(dtype=x.dtype)
        modulation = self.modulation.to(dtype=x.dtype)
        e = (modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        # Used by OmniFlow trainer: it sets `model.dit.gradient_checkpointing = True`
        # when `trainer_args.gradient_checkpointing: true`.
        self.gradient_checkpointing = False

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks (use DiTBlock for YAML compatibility)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # rope buffers (avoid register_buffer to keep dtype stable under .to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        self.init_weights()

    def forward(self, x, t, context, seq_len, y=None):
        if self.model_type == "i2v":
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Cast large modulation tensors back to model dtype early to reduce peak memory.
        e = e.to(dtype=x.dtype)
        e0 = e0.to(dtype=x.dtype)

        # context
        context_lens = None
        context_in = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        context_in = context_in.to(dtype=self.text_embedding[0].weight.dtype)
        context = self.text_embedding(context_in)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        freqs = kwargs["freqs"]
        context_lens_ = kwargs["context_lens"]
        seq_lens_ = kwargs["seq_lens"]
        grid_sizes_ = kwargs["grid_sizes"]
        e0_ = kwargs["e"]
        context_ = kwargs["context"]

        for block in self.blocks:
            # Match `wan_new` behavior: block-level activation checkpointing.
            # This is critical for long sequences (e.g., Wan2.1 latents), otherwise
            # activations can exceed GPU memory even when `wan_new` fits.
            if self.training and getattr(self, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(x_in, e_in, ctx_in):
                        return module(
                            x_in,
                            e=e_in,
                            seq_lens=seq_lens_,
                            grid_sizes=grid_sizes_,
                            freqs=freqs,
                            context=ctx_in,
                            context_lens=context_lens_,
                        )

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    e0_,
                    context_,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        # IMPORTANT:
        # Returning fp32 here dramatically increases activation/grad memory and slows
        # training (DiffSynth returns bf16 here). Keep dtype consistent with model.
        return x

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

