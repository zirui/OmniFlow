# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compile-friendly Primus-Turbo MXFP4 linear kernels for FLUX."""

import os

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


@torch.library.custom_op(
    "omniflow::quantize_mxfp4_colwise", mutates_args=(), device_types="cuda"
)
def _quantize_mxfp4_colwise(
    x: torch.Tensor,
    use_2d_block: bool,
    use_rht: bool,
    shuffle_scale: bool,
    shuffle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4(
        x,
        _FP4_DTYPE,
        0,
        _MXFP4_PADDING_ALIGN_SIZE,
        use_2d_block,
        False,
        use_rht,
        shuffle_scale,
        shuffle,
    )


@_quantize_mxfp4_colwise.register_fake
def _quantize_mxfp4_colwise_fake(
    x: torch.Tensor,
    use_2d_block: bool,
    use_rht: bool,
    shuffle_scale: bool,
    shuffle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del use_2d_block, use_rht, shuffle
    rows, columns = x.shape
    rows_padded = _cdiv(rows, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    scale_rows = _cdiv(columns, 256) * 256 if shuffle_scale else columns
    scale_columns = _cdiv(rows_padded, _MXFP4_BLOCK_SIZE)
    if shuffle_scale:
        scale_columns = _cdiv(scale_columns, 8) * 8
    return (
        torch.empty(columns, rows_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(scale_rows, scale_columns, dtype=torch.uint8, device=x.device).view(
            torch.float8_e8m0fnu
        ),
    )


_quantize_mxfp4_colwise.register_autograd(
    lambda ctx, *grads: (None,) * 5,
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op(
    "omniflow::quantize_mxfp8_rowwise_mxfp4_colwise",
    mutates_args=(),
    device_types="cuda",
)
def _quantize_mxfp8_rowwise_mxfp4_colwise(
    x: torch.Tensor,
    rowwise_fp8_use_2d_block: bool,
    colwise_fp4_use_2d_block: bool,
    colwise_fp4_use_rht: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8_rowwise_mxfp4_colwise(
        x,
        rowwise_fp8_use_2d_block,
        colwise_fp4_use_2d_block,
        colwise_fp4_use_rht,
        shuffle_colwise_scale,
        shuffle_colwise,
    )


@_quantize_mxfp8_rowwise_mxfp4_colwise.register_fake
def _quantize_mxfp8_rowwise_mxfp4_colwise_fake(
    x: torch.Tensor,
    rowwise_fp8_use_2d_block: bool,
    colwise_fp4_use_2d_block: bool,
    colwise_fp4_use_rht: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        rowwise_fp8_use_2d_block,
        colwise_fp4_use_2d_block,
        colwise_fp4_use_rht,
        shuffle_colwise,
    )
    rows, columns = x.shape
    rows_padded = _cdiv(rows, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    columns_padded = _cdiv(columns, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    colwise_scale_rows = _cdiv(columns, 256) * 256 if shuffle_colwise_scale else columns
    colwise_scale_columns = _cdiv(rows_padded, _MXFP4_BLOCK_SIZE)
    if shuffle_colwise_scale:
        colwise_scale_columns = _cdiv(colwise_scale_columns, 8) * 8
    return (
        torch.empty(rows, columns_padded, dtype=_MXFP8_DTYPE, device=x.device),
        torch.empty(
            rows,
            _cdiv(columns_padded, _MXFP4_BLOCK_SIZE),
            dtype=torch.float8_e8m0fnu,
            device=x.device,
        ),
        torch.empty(columns, rows_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(
            colwise_scale_rows,
            colwise_scale_columns,
            dtype=torch.uint8,
            device=x.device,
        ).view(torch.float8_e8m0fnu),
    )


_quantize_mxfp8_rowwise_mxfp4_colwise.register_autograd(
    lambda ctx, *grads: (None,) * 6,
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


@torch.library.custom_op("omniflow::dequantize_mxfp4_rowwise", mutates_args=(), device_types="cuda")
def _dequantize_mxfp4_rowwise(
    x: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    return torch.ops.primus_turbo_cpp_extension.dequantize_mxfp4(
        x, scale, 1, _MXFP4_BLOCK_SIZE, out_dtype
    )


@_dequantize_mxfp4_rowwise.register_fake
def _dequantize_mxfp4_rowwise_fake(
    x: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    del scale
    return torch.empty(x.shape[0], x.shape[1] * 2, dtype=out_dtype, device=x.device)


_dequantize_mxfp4_rowwise.register_autograd(
    lambda ctx, *grads: (None, None, None),
    setup_context=lambda ctx, inputs, output: None,
)


def _quantize_input(
    input_2d: torch.Tensor, preshuffle: bool, forward_rht: bool = False
):
    return _quantize_mxfp4_dual(
        input_2d,
        False,
        False,
        forward_rht,
        False,
        False,
        True,
        preshuffle,
        False,
        preshuffle,
        preshuffle,
    )


def _quantize_weight(
    weight: torch.Tensor, preshuffle: bool, forward_rht: bool = False
):
    return _quantize_mxfp4_dual(
        weight,
        True,
        False,
        forward_rht,
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
    def forward(
        input,
        weight,
        preshuffle,
        use_gradient_sr,
        use_forward_rht,
        activation_residual_mode,
    ):
        input_2d = input.reshape(-1, input.shape[-1])
        a, a_scale, a_t, a_t_scale = _quantize_input(
            input_2d, preshuffle, use_forward_rht
        )
        b, b_scale, b_t, b_t_scale = _quantize_weight(
            weight, preshuffle, use_forward_rht
        )
        output = _gemm_fp4(a, a_scale, False, b, b_scale, True, input.dtype, preshuffle)
        if activation_residual_mode:
            q_input, q_scale, _, _ = _quantize_input(input_2d, False)
            q_input = _dequantize_mxfp4_rowwise(q_input, q_scale, input.dtype)
            residual = input_2d - q_input
            if activation_residual_mode == 1:
                correction = torch.nn.functional.linear(residual, weight)
            elif activation_residual_mode == 2:
                r, r_scale = _quantize_mxfp8_rowwise(residual, False)
                w, w_scale = _quantize_mxfp8_rowwise(weight, True)
                correction = gemm_fp8_impl(
                    r,
                    r_scale,
                    False,
                    w,
                    w_scale,
                    True,
                    input.dtype,
                    False,
                    granularity=_GRANULARITY,
                    default_backend=BackendType.FLYDSL.value,
                )
            else:
                r, r_scale, _, _ = _quantize_input(residual, preshuffle)
                correction = _gemm_fp4(
                    r, r_scale, False, b, b_scale, True, input.dtype, preshuffle
                )
            output = output + correction
        return (
            output,
            a_t.view(torch.uint8),
            a_t_scale.view(torch.uint8),
            b_t.view(torch.uint8),
            b_t_scale.view(torch.uint8),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, _, preshuffle, use_gradient_sr, _, _ = inputs
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
        grad_input, grad_weight = _mxfp4_backward(
            ctx, grad_output, *ctx.saved_tensors
        )
        return grad_input, grad_weight, None, None, None, None


def _mxfp4_backward(ctx, grad_output, a_t, a_t_scale, b_t, b_t_scale):
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    gradient_2d = grad_output.reshape(-1, grad_output.shape[-1])
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
        a, a_scale, a_t, a_t_scale = _quantize_mxfp8_rowwise_mxfp4_colwise(
            input_2d, False, False, True, preshuffle, preshuffle
        )
        b, b_scale, b_t, b_t_scale = _quantize_mxfp8_rowwise_mxfp4_colwise(
            weight, True, True, False, preshuffle, preshuffle
        )
        output = gemm_fp8_impl(
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
        grad_input, grad_weight = _mxfp4_backward(
            ctx, grad_output, *ctx.saved_tensors
        )
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
        input_2d = input.reshape(-1, input.shape[-1])
        a_t, a_t_scale = _quantize_mxfp4_colwise(
            input_2d, False, True, ctx.preshuffle, ctx.preshuffle
        )
        b_t, b_t_scale = _quantize_mxfp4_colwise(
            weight, True, False, ctx.preshuffle, ctx.preshuffle
        )
        grad_input, grad_weight = _mxfp4_backward(
            ctx, grad_output, a_t, a_t_scale, b_t, b_t_scale
        )
        return grad_input, grad_weight, None, None


_FORWARD_FUNCTIONS = {
    "bf16": _BF16Forward,
    "mxfp4": _MXFP4Forward,
    "mxfp8": _MXFP8Forward,
}

_MXFP4_CAPTURE_STEP = -1
_MXFP4_CAPTURED: set[tuple[int, str]] = set()


def set_mxfp4_capture_step(step: int) -> None:
    """Set the optimizer step associated with subsequent forward captures."""
    global _MXFP4_CAPTURE_STEP
    _MXFP4_CAPTURE_STEP = step


def _save_mxfp4_capture(input: torch.Tensor, weight: torch.Tensor, module_name: str) -> None:
    steps = {
        int(value)
        for value in os.getenv("FLUX_MXFP4_CAPTURE_STEPS", "").split(",")
        if value
    }
    key = (_MXFP4_CAPTURE_STEP, module_name)
    if _MXFP4_CAPTURE_STEP not in steps or key in _MXFP4_CAPTURED:
        return
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        output_dir = os.environ["FLUX_MXFP4_CAPTURE_DIR"]
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "step": _MXFP4_CAPTURE_STEP,
                "module": module_name,
                "input": input.reshape(-1, input.shape[-1])[:1024].detach().cpu(),
                "weight": weight.detach().cpu(),
            },
            os.path.join(
                output_dir,
                f"step{_MXFP4_CAPTURE_STEP}_{module_name.replace('.', '_')}.pt",
            ),
        )
    _MXFP4_CAPTURED.add(key)


@torch.library.custom_op("omniflow::capture_mxfp4_linear", mutates_args=(), device_types="cuda")
def _capture_mxfp4_linear(
    input: torch.Tensor, weight: torch.Tensor, module_name: str
) -> torch.Tensor:
    _save_mxfp4_capture(input, weight, module_name)
    return input.clone()


@_capture_mxfp4_linear.register_fake
def _capture_mxfp4_linear_fake(
    input: torch.Tensor, weight: torch.Tensor, module_name: str
) -> torch.Tensor:
    del weight, module_name
    return torch.empty_like(input)


_capture_mxfp4_linear.register_autograd(
    lambda ctx, grad: (grad, None, None),
    setup_context=lambda ctx, inputs, output: None,
)


class MXFP4Linear(torch.nn.Module):
    """Linear with recipe-selected forward precision and MXFP4 backward."""

    backward_precision = "mxfp4"

    def __init__(
        self,
        linear: torch.nn.Linear,
        forward_precision: str,
        *,
        forward_hadamard: bool = False,
        activation_residual_mode: int = 0,
        eval_bf16: bool = False,
        gradient_stochastic_rounding: bool = False,
        fqn: str = "",
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.forward_precision = forward_precision
        self.forward_hadamard = forward_hadamard
        self.activation_residual_mode = activation_residual_mode
        self.eval_bf16 = eval_bf16
        self.gradient_stochastic_rounding = gradient_stochastic_rounding
        capture_modules = set(
            filter(None, os.getenv("FLUX_MXFP4_CAPTURE_MODULES", "").split(","))
        )
        self._capture_name = fqn if fqn in capture_modules else ""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.eval_bf16 and not self.training:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        shape = input.shape[:-1]
        if self._capture_name:
            input = _capture_mxfp4_linear(input, self.weight, self._capture_name)
        function = _FORWARD_FUNCTIONS[self.forward_precision]
        if self.forward_precision == "mxfp4":
            result = function.apply(
                input,
                self.weight,
                True,
                self.gradient_stochastic_rounding,
                self.forward_hadamard,
                self.activation_residual_mode,
            )
            output = result[0]
        else:
            result = function.apply(
                input, self.weight, True, self.gradient_stochastic_rounding
            )
            output = result[0] if self.forward_precision == "mxfp8" else result
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape, self.out_features)

    def switch_forward_to_bf16(self) -> bool:
        """Heal an MXFP4 forward while preserving its MXFP4 backward."""
        if self.forward_precision != "mxfp4":
            return False
        self.forward_precision = "bf16"
        self.forward_hadamard = False
        self.activation_residual_mode = 0
        return True
