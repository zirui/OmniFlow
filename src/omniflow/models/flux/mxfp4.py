# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compile-friendly Primus-Turbo MXFP4 linear kernels for FLUX."""

import torch
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

_MXFP4_BLOCK_SIZE = 32
_MXFP4_PADDING_ALIGN_SIZE = 128
_FP4_DTYPE = torch.float4_e2m1fn_x2
_MXFP8_DTYPE = torch.float8_e4m3fn
_GRANULARITY = ScalingGranularity.MX_BLOCKWISE.value
_FP4_BACKEND = BackendType.HIPBLASLT.value


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@torch.library.custom_op("omniflow::quantize_mxfp4_dual", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_dual(
    x: torch.Tensor,
    rowwise_use_2d_block: bool,
    rowwise_use_sr: bool,
    rowwise_use_rht: bool,
    colwise_use_2d_block: bool,
    colwise_use_sr: bool,
    colwise_use_rht: bool,
    shuffle_rowwise_scale: bool,
    shuffle_rowwise: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x,
        _FP4_DTYPE,
        _MXFP4_PADDING_ALIGN_SIZE,
        rowwise_use_2d_block,
        rowwise_use_sr,
        rowwise_use_rht,
        colwise_use_2d_block,
        colwise_use_sr,
        colwise_use_rht,
        shuffle_rowwise_scale,
        shuffle_rowwise,
        shuffle_colwise_scale,
        shuffle_colwise,
    )


@_quantize_mxfp4_dual.register_fake
def _quantize_mxfp4_dual_fake(
    x: torch.Tensor,
    rowwise_use_2d_block: bool,
    rowwise_use_sr: bool,
    rowwise_use_rht: bool,
    colwise_use_2d_block: bool,
    colwise_use_sr: bool,
    colwise_use_rht: bool,
    shuffle_rowwise_scale: bool,
    shuffle_rowwise: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        rowwise_use_2d_block,
        rowwise_use_sr,
        rowwise_use_rht,
        colwise_use_2d_block,
        colwise_use_sr,
        colwise_use_rht,
        shuffle_rowwise,
        shuffle_colwise,
    )
    rows, columns = x.shape
    rows_padded = _cdiv(rows, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    columns_padded = _cdiv(columns, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    rowwise_scale_rows = _cdiv(rows, 256) * 256 if shuffle_rowwise_scale else rows
    rowwise_scale_columns = _cdiv(_cdiv(columns_padded, _MXFP4_BLOCK_SIZE), 8) * 8
    if not shuffle_rowwise_scale:
        rowwise_scale_columns = _cdiv(columns_padded, _MXFP4_BLOCK_SIZE)
    colwise_scale_rows = _cdiv(columns, 256) * 256 if shuffle_colwise_scale else columns
    colwise_scale_columns = _cdiv(_cdiv(rows_padded, _MXFP4_BLOCK_SIZE), 8) * 8
    if not shuffle_colwise_scale:
        colwise_scale_columns = _cdiv(rows_padded, _MXFP4_BLOCK_SIZE)
    return (
        torch.empty(rows, columns_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(
            rowwise_scale_rows,
            rowwise_scale_columns,
            dtype=torch.uint8,
            device=x.device,
        ).view(torch.float8_e8m0fnu),
        torch.empty(columns, rows_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(
            colwise_scale_rows,
            colwise_scale_columns,
            dtype=torch.uint8,
            device=x.device,
        ).view(torch.float8_e8m0fnu),
    )


_quantize_mxfp4_dual.register_autograd(
    lambda ctx, *grads: (None,) * 11,
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op("omniflow::quantize_mxfp8_rowwise", mutates_args=(), device_types="cuda")
def _quantize_mxfp8_rowwise(
    x: torch.Tensor, use_2d_block: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(
        x, _MXFP8_DTYPE, 1, _MXFP4_PADDING_ALIGN_SIZE, use_2d_block, False, False
    )


@_quantize_mxfp8_rowwise.register_fake
def _quantize_mxfp8_rowwise_fake(
    x: torch.Tensor, use_2d_block: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    del use_2d_block
    rows, columns = x.shape
    columns_padded = _cdiv(columns, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    return (
        torch.empty(rows, columns_padded, dtype=_MXFP8_DTYPE, device=x.device),
        torch.empty(
            rows,
            _cdiv(columns_padded, _MXFP4_BLOCK_SIZE),
            dtype=torch.float8_e8m0fnu,
            device=x.device,
        ),
    )


_quantize_mxfp8_rowwise.register_autograd(
    lambda ctx, *grads: (None, None),
    setup_context=lambda ctx, inputs, output: None,
)


def _quantize_input(input_2d: torch.Tensor, preshuffle: bool):
    return _quantize_mxfp4_dual(
        input_2d,
        False,
        False,
        False,
        False,
        False,
        True,
        preshuffle,
        False,
        preshuffle,
        preshuffle,
    )


def _quantize_weight(weight: torch.Tensor, preshuffle: bool):
    return _quantize_mxfp4_dual(
        weight,
        True,
        False,
        False,
        True,
        False,
        False,
        preshuffle,
        preshuffle,
        preshuffle,
        preshuffle,
    )


def _quantize_gradient(gradient_2d: torch.Tensor, preshuffle: bool, use_sr: bool):
    return _quantize_mxfp4_dual(
        gradient_2d,
        False,
        use_sr,
        False,
        False,
        use_sr,
        True,
        preshuffle,
        False,
        preshuffle,
        False,
    )


def _gemm_fp4(a, a_scale, trans_a, b, b_scale, trans_b, out_dtype, preshuffle):
    return gemm_fp4_impl(
        a,
        a_scale,
        trans_a,
        b,
        b_scale,
        trans_b,
        out_dtype,
        False,
        granularity=_GRANULARITY,
        default_backend=_FP4_BACKEND,
        preshuffled=preshuffle,
    )


class _MXFP4Forward(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, preshuffle, use_gradient_sr):
        input_2d = input.reshape(-1, input.shape[-1])
        a, a_scale, a_t, a_t_scale = _quantize_input(input_2d, preshuffle)
        b, b_scale, b_t, b_t_scale = _quantize_weight(weight, preshuffle)
        output = _gemm_fp4(a, a_scale, False, b, b_scale, True, input.dtype, preshuffle)
        return (
            output,
            a_t.view(torch.uint8),
            a_t_scale.view(torch.uint8),
            b_t.view(torch.uint8),
            b_t_scale.view(torch.uint8),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, _, preshuffle, use_gradient_sr = inputs
        ctx.preshuffle = preshuffle
        ctx.use_gradient_sr = use_gradient_sr
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        _, a_t, a_t_scale, b_t, b_t_scale = output
        ctx.save_for_backward(
            a_t.view(_FP4_DTYPE),
            a_t_scale.view(torch.float8_e8m0fnu),
            b_t.view(_FP4_DTYPE),
            b_t_scale.view(torch.float8_e8m0fnu),
        )
        ctx.mark_non_differentiable(a_t, a_t_scale, b_t, b_t_scale)

    @staticmethod
    def backward(ctx, grad_output, *_):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        gradient_2d = grad_output.reshape(-1, grad_output.shape[-1])
        a_t, a_t_scale, b_t, b_t_scale = ctx.saved_tensors
        g, g_scale, g_t, g_t_scale = _quantize_gradient(
            gradient_2d, ctx.preshuffle, ctx.use_gradient_sr
        )
        grad_input = _gemm_fp4(
            g, g_scale, False, b_t, b_t_scale, True, ctx.out_dtype, ctx.preshuffle
        ).reshape(ctx.orig_shape)
        grad_weight = _gemm_fp4(
            g_t, g_t_scale, False, a_t, a_t_scale, True, ctx.out_dtype, ctx.preshuffle
        )
        return grad_input, grad_weight, None, None


def _mxfp4_backward(ctx, grad_output, input, weight):
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    gradient_2d = grad_output.reshape(-1, grad_output.shape[-1])
    input_2d = input.reshape(-1, input.shape[-1])
    _, _, a_t, a_t_scale = _quantize_input(input_2d, ctx.preshuffle)
    _, _, b_t, b_t_scale = _quantize_weight(weight, ctx.preshuffle)
    g, g_scale, g_t, g_t_scale = _quantize_gradient(
        gradient_2d, ctx.preshuffle, ctx.use_gradient_sr
    )
    grad_input = _gemm_fp4(
        g, g_scale, False, b_t, b_t_scale, True, ctx.out_dtype, ctx.preshuffle
    ).reshape(ctx.orig_shape)
    grad_weight = _gemm_fp4(
        g_t, g_t_scale, False, a_t, a_t_scale, True, ctx.out_dtype, ctx.preshuffle
    )
    return grad_input, grad_weight


class _MXFP8Forward(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, preshuffle, use_gradient_sr):
        input_2d = input.reshape(-1, input.shape[-1])
        a, a_scale = _quantize_mxfp8_rowwise(input_2d, False)
        b, b_scale = _quantize_mxfp8_rowwise(weight, True)
        return gemm_fp8_impl(
            a,
            a_scale,
            False,
            b,
            b_scale,
            True,
            input.dtype,
            False,
            granularity=_GRANULARITY,
            default_backend=BackendType.FLYDSL.value,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, preshuffle, use_gradient_sr = inputs
        ctx.preshuffle = preshuffle
        ctx.use_gradient_sr = use_gradient_sr
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        ctx.save_for_backward(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = _mxfp4_backward(ctx, grad_output, input, weight)
        return grad_input, grad_weight, None, None


class _BF16Forward(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, preshuffle, use_gradient_sr):
        return torch.nn.functional.linear(input.reshape(-1, input.shape[-1]), weight)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, preshuffle, use_gradient_sr = inputs
        ctx.preshuffle = preshuffle
        ctx.use_gradient_sr = use_gradient_sr
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        ctx.save_for_backward(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = _mxfp4_backward(ctx, grad_output, input, weight)
        return grad_input, grad_weight, None, None


_FORWARD_FUNCTIONS = {
    "bf16": _BF16Forward,
    "mxfp4": _MXFP4Forward,
    "mxfp8": _MXFP8Forward,
}


class MXFP4Linear(torch.nn.Module):
    """Linear with recipe-selected forward precision and MXFP4 backward."""

    backward_precision = "mxfp4"

    def __init__(self, linear: torch.nn.Linear, forward_precision: str) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.forward_precision = forward_precision
        self._function = _FORWARD_FUNCTIONS[forward_precision]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shape = input.shape[:-1]
        result = self._function.apply(input, self.weight, True, False)
        output = result[0] if self.forward_precision == "mxfp4" else result
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape, self.out_features)
