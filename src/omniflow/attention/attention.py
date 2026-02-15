"""
Unified attention entry points.

Public API:
- `attention(q, k, v, ...)`        : Wan2.2-style, q/k/v shape [B, L, N, D]
- `attention_fused(q, k, v, ...)`  : Legacy Wan-style, q/k/v shape [B, S, N*D]
"""

from __future__ import annotations

import warnings

import torch

FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
flash_attn_interface = None
flash_attn = None

try:
    import flash_attn_interface as _flash_attn_interface  # type: ignore

    flash_attn_interface = _flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn as _flash_attn  # type: ignore

    flash_attn = _flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception:
    FLASH_ATTN_2_AVAILABLE = False


__all__ = [
    "FLASH_ATTN_2_AVAILABLE",
    "FLASH_ATTN_3_AVAILABLE",
    "attention",
    "attention_fused",
    "flash_attention",
    "get_attention_backend",
    "set_attention_backend",
]


# ---------------------------------------------------------------------------
# Global backend config (set once at startup, e.g. from YAML)
# ---------------------------------------------------------------------------
_ATTENTION_BACKEND: str = "auto"


_VALID_BACKENDS = ("auto", "sdpa", "flex_attention", "flash_attn2", "flash_attn3")


def set_attention_backend(backend: str) -> None:
    """
    Set global attention backend.

    Accepted values (case-insensitive):
    - "auto" | "sdpa" | "flash_attn2" | "flash_attn3"
    """
    global _ATTENTION_BACKEND
    backend = (backend or "").strip().lower()
    if backend not in _VALID_BACKENDS:
        hint = ""
        if backend.startswith("flash_atten"):
            hint = " (did you mean 'flash_attn2' / 'flash_attn3'?)"
        raise ValueError(
            f"attention_backend must be one of {_VALID_BACKENDS}, got '{backend}'{hint}"
        )
    _ATTENTION_BACKEND = backend


def get_attention_backend() -> str:
    return _ATTENTION_BACKEND


def _resolve_flash_version(device_type: str) -> int | None:
    """
    Returns:
        - None: do not use flash attention (use SDPA)
        - 2 or 3: use flash attention, prefer that version (3 may fall back to 2 in `flash_attention`)
    """
    backend = _ATTENTION_BACKEND
    if backend in ("sdpa", "flex_attention"):
        return None

    if device_type != "cuda":
        # auto: CPU should use SDPA
        if backend == "auto":
            return None
        # explicit flash backend: error (prevents silent slow CPU path)
        raise RuntimeError(f"attention_backend='{backend}' requires CUDA tensors.")

    # CUDA path
    if backend == "auto":
        if FLASH_ATTN_3_AVAILABLE:
            return 3
        if FLASH_ATTN_2_AVAILABLE:
            return 2
        return None

    if backend == "flash_attn2":
        if not FLASH_ATTN_2_AVAILABLE:
            raise RuntimeError("attention_backend='flash_attn2' but flash-attn 2 is not available.")
        return 2

    # "flash_attn3"
    if not FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError("attention_backend='flash_attn3' but flash-attn 3 is not available.")
    return 3


# ---------------------------------------------------------------------------
# Flash Attention implementation (varlen + fixed-length fast path)
# ---------------------------------------------------------------------------
def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Fast path (fixed-length, no padding mask):
    # Prefer flash_attn_func over flash_attn_varlen_func to reduce overhead.
    if q_lens is None and k_lens is None:
        qh = half(q)
        kh = half(k)
        vh = half(v)
        qh = qh.to(vh.dtype)
        kh = kh.to(vh.dtype)
        if q_scale is not None:
            qh = qh * q_scale

        # apply attention
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_func(qh, kh, vh)
            if isinstance(x, tuple):
                x = x[0]
        else:
            assert FLASH_ATTN_2_AVAILABLE
            # flash_attn_func signature differs across versions; be conservative
            try:
                x = flash_attn.flash_attn_func(
                    qh,
                    kh,
                    vh,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                )
            except TypeError:
                x = flash_attn.flash_attn_func(qh, kh, vh)
        return x.type(out_dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens, strict=False)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens, strict=False)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens, strict=False)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            "Flash attention 3 is not available, use flash attention 2 instead.",
            stacklevel=2,
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(
                q.device, non_blocking=True
            ),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(
                q.device, non_blocking=True
            ),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(
                q.device, non_blocking=True
            ),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(
                q.device, non_blocking=True
            ),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    Unified attention for Wan2.2-style tensors.

    q: [B, Lq, N, D]
    k: [B, Lk, N, D]
    v: [B, Lk, N, D]
    """
    resolved = _resolve_flash_version(q.device.type)
    if resolved is not None:
        version = resolved
        # Only "auto" allows call-site override (useful for debugging).
        if _ATTENTION_BACKEND == "auto" and fa_version is not None:
            version = fa_version
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )

    # FlexAttention path (native PyTorch, requires torch.compile for perf)
    if _ATTENTION_BACKEND == "flex_attention":
        from .flex import flex_attention_fn

        return flex_attention_fn(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
        )

    # SDPA fallback (also used for CPU).
    # Currently assumes *uniform-length* training sequences, so we do NOT support padding masks on the SDPA path yet
    if q_lens is not None or k_lens is not None:
        warnings.warn("Padding mask is disabled when using scaled_dot_product_attention.", stacklevel=2)
    out_dtype = q.dtype
    q_ = q.transpose(1, 2).to(dtype)
    k_ = k.transpose(1, 2).to(dtype)
    v_ = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q_ = q_ * q_scale
    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous().to(out_dtype)


def attention_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
):
    """
    Legacy Wan fused-head attention.

    q/k/v: [B, S, num_heads * head_dim]
    returns: [B, S, num_heads * head_dim]
    """
    b, s, c = q.shape
    if c % num_heads != 0:
        raise ValueError(f"hidden dim {c} not divisible by num_heads {num_heads}")
    d = c // num_heads

    q4d = q.view(b, s, num_heads, d)
    k4d = k.view(b, -1, num_heads, d)
    v4d = v.view(b, -1, num_heads, d)

    # For fused calls, q_lens/k_lens are not used (fixed length).
    out = attention(
        q=q4d,
        k=k4d,
        v=v4d,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        dtype=q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
    )
    return out.flatten(2)