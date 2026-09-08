from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import pytest
import torch


def _load_mxfp4(monkeypatch):
    loaded = sys.modules.get("omniflow.models.flux.mxfp4")
    if loaded is not None:
        return loaded
    backend = types.ModuleType("primus_turbo.pytorch.core.backend")
    backend.BackendType = types.SimpleNamespace(
        HIPBLASLT=types.SimpleNamespace(value="hipblaslt"),
        FLYDSL=types.SimpleNamespace(value="flydsl"),
    )
    low_precision = types.ModuleType("primus_turbo.pytorch.core.low_precision")
    low_precision.ScalingGranularity = types.SimpleNamespace(MX_BLOCKWISE=types.SimpleNamespace(value="mx"))
    gemm_fp4 = types.ModuleType("primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl")
    gemm_fp4.gemm_fp4_impl = lambda *args, **kwargs: None
    gemm_fp8 = types.ModuleType("primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl")
    gemm_fp8.gemm_fp8_impl = lambda *args, **kwargs: None
    for module in (backend, low_precision, gemm_fp4, gemm_fp8):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return importlib.import_module("omniflow.models.flux.mxfp4")


def test_mxfp4_quantizer_custom_ops_have_compile_schemas_and_fake_shapes(monkeypatch):
    mxfp4 = _load_mxfp4(monkeypatch)

    assert str(mxfp4._quantize_mxfp4_colwise._schema) == (
        "(Tensor x, bool use_2d_block, bool use_rht, bool shuffle_scale, bool shuffle) -> (Tensor, Tensor)"
    )
    assert str(mxfp4._quantize_mxfp8_rowwise_mxfp4_colwise._schema) == (
        "(Tensor x, bool rowwise_fp8_use_2d_block, bool colwise_fp4_use_2d_block, "
        "bool colwise_fp4_use_rht, bool shuffle_colwise_scale, bool shuffle_colwise) "
        "-> (Tensor, Tensor, Tensor, Tensor)"
    )

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(96, 160, device="cuda")
        mixed = mxfp4._quantize_mxfp8_rowwise_mxfp4_colwise(x, False, True, True, True, True)
        assert [(tuple(t.shape), t.dtype) for t in mixed] == [
            ((96, 256), torch.float8_e4m3fn),
            ((96, 8), torch.float8_e8m0fnu),
            ((160, 64), torch.float4_e2m1fn_x2),
            ((256, 8), torch.float8_e8m0fnu),
        ]
        colwise = mxfp4._quantize_mxfp4_colwise(x, False, True, False, False)
        assert [(tuple(t.shape), t.dtype) for t in colwise] == [
            ((160, 64), torch.float4_e2m1fn_x2),
            ((160, 4), torch.float8_e8m0fnu),
        ]


def test_mxfp4_quantizer_wrappers_call_raw_ops_and_have_nondifferentiable_autograd(
    monkeypatch,
):
    mxfp4 = _load_mxfp4(monkeypatch)
    x = torch.ones(2, 32)
    calls = []

    def raw_colwise(*args):
        calls.append(("colwise", args))
        return (x, x)

    def raw_mixed(*args):
        calls.append(("mixed", args))
        return (x, x, x, x)

    monkeypatch.setattr(
        torch.ops.primus_turbo_cpp_extension,
        "quantize_mxfp4",
        raw_colwise,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.primus_turbo_cpp_extension,
        "quantize_mxfp8_rowwise_mxfp4_colwise",
        raw_mixed,
        raising=False,
    )

    mxfp4._quantize_mxfp4_colwise._init_fn(x, True, False, True, True)
    mxfp4._quantize_mxfp8_rowwise_mxfp4_colwise._init_fn(x, False, True, False, True, True)

    assert calls[0][0] == "colwise"
    assert calls[0][1][1:] == (
        torch.float4_e2m1fn_x2,
        0,
        128,
        True,
        False,
        False,
        True,
        True,
    )
    assert calls[1] == ("mixed", (x, False, True, False, True, True))
    assert mxfp4._quantize_mxfp4_colwise._backward_fn(None, None, None) == (None,) * 5
    assert mxfp4._quantize_mxfp8_rowwise_mxfp4_colwise._backward_fn(None, None, None, None, None) == (None,) * 6


def test_mxfp8_forward_saves_fused_colwise_outputs_for_backward(monkeypatch):
    mxfp4 = _load_mxfp4(monkeypatch)
    input = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    calls = []

    def fused(value, *options):
        calls.append(options)
        colwise = torch.zeros(value.shape[1], 64, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        scale = torch.zeros(value.shape[1], 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return value, torch.empty(0), colwise, scale

    saved = []

    def backward(ctx, grad_output, *quantized):
        saved.extend(quantized)
        return torch.zeros(ctx.orig_shape), torch.zeros(3, 4)

    monkeypatch.setattr(mxfp4, "_quantize_mxfp8_rowwise_mxfp4_colwise", fused)
    monkeypatch.setattr(mxfp4, "_mxfp4_backward", backward)
    monkeypatch.setattr(
        mxfp4,
        "gemm_fp8_impl",
        lambda a, a_scale, trans_a, b, b_scale, trans_b, *args, **kwargs: a @ b.T,
    )
    monkeypatch.setattr(
        mxfp4,
        "_quantize_mxfp8_rowwise",
        lambda *args: pytest.fail("separate MXFP8 quantization must not run"),
    )
    monkeypatch.setattr(
        mxfp4,
        "_quantize_mxfp4_dual",
        lambda *args: pytest.fail("MXFP8 backward must not requantize BF16 tensors"),
    )

    packed = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: packed.append(tensor.dtype) or tensor,
        lambda tensor: tensor,
    ):
        output = mxfp4._MXFP8Forward.apply(input, weight, True, False)[0]
        output.sum().backward()

    assert packed == [
        torch.float4_e2m1fn_x2,
        torch.float8_e8m0fnu,
        torch.float4_e2m1fn_x2,
        torch.float8_e8m0fnu,
    ]
    assert calls == [
        (False, False, True, True, True),
        (True, True, False, True, True),
    ]
    assert [tensor.dtype for tensor in saved] == [
        torch.float4_e2m1fn_x2,
        torch.float8_e8m0fnu,
        torch.float4_e2m1fn_x2,
        torch.float8_e8m0fnu,
    ]


def test_bf16_backward_only_quantizes_colwise_mxfp4(monkeypatch):
    mxfp4 = _load_mxfp4(monkeypatch)
    input = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    calls = []

    def colwise(value, *options):
        calls.append((value.shape, options))
        return value, torch.empty(0)

    monkeypatch.setattr(mxfp4, "_quantize_mxfp4_colwise", colwise)
    monkeypatch.setattr(
        mxfp4,
        "_quantize_mxfp4_dual",
        lambda *args: pytest.fail("BF16 backward must not request rowwise MXFP4"),
    )
    monkeypatch.setattr(
        mxfp4,
        "_mxfp4_backward",
        lambda ctx, grad_output, *quantized: (
            torch.zeros(ctx.orig_shape),
            torch.zeros(3, 4),
        ),
    )

    packed = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: packed.append(tensor) or tensor,
        lambda tensor: tensor,
    ):
        mxfp4._BF16Forward.apply(input, weight, True, False).sum().backward()

    assert packed == [input, weight]
    assert calls == [
        (torch.Size([2, 4]), (False, True, True, True)),
        (torch.Size([3, 4]), (True, False, True, True)),
    ]


def test_mxfp4_paired_forward_hadamard_only_changes_rowwise_quantization(monkeypatch):
    mxfp4 = _load_mxfp4(monkeypatch)
    calls = []
    monkeypatch.setattr(mxfp4, "_quantize_mxfp4_dual", lambda *args: calls.append(args) or ())
    tensor = torch.ones(2, 4)

    mxfp4._quantize_input(tensor, True, True)
    mxfp4._quantize_weight(tensor, True, True)

    assert calls[0][3] is True
    assert calls[1][3] is True
    assert calls[0][6] is True
    assert calls[1][6] is False


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_mxfp4_activation_residual_uses_input_quantization_error(monkeypatch, mode):
    mxfp4 = _load_mxfp4(monkeypatch)
    input = torch.tensor([[3.0]])
    weight = torch.tensor([[2.0]])
    input_calls = 0
    forward_weight = torch.tensor([[2.0]])
    used_weights = []

    def quantize_input(value, preshuffle, forward_rht=False):
        nonlocal input_calls
        input_calls += 1
        quantized = value - 1 if input_calls <= 2 else value
        marker = torch.zeros(1, dtype=torch.uint8)
        return quantized, marker, marker, marker

    def quantize_weight(value, preshuffle, forward_rht=False):
        marker = torch.zeros(1, dtype=torch.uint8)
        return forward_weight, marker, marker, marker

    def gemm_fp4(a, a_scale, trans_a, b, b_scale, trans_b, out_dtype, preshuffle):
        used_weights.append(b)
        return a @ weight.T

    monkeypatch.setattr(mxfp4, "_quantize_input", quantize_input)
    monkeypatch.setattr(mxfp4, "_quantize_weight", quantize_weight)
    monkeypatch.setattr(mxfp4, "_dequantize_mxfp4_rowwise", lambda value, scale, dtype: value)
    monkeypatch.setattr(mxfp4, "_gemm_fp4", gemm_fp4)
    monkeypatch.setattr(mxfp4, "_quantize_mxfp8_rowwise", lambda value, use_2d: (value, object()))
    monkeypatch.setattr(
        mxfp4,
        "gemm_fp8_impl",
        lambda a, a_scale, trans_a, b, b_scale, trans_b, *args, **kwargs: a @ b.T,
    )

    output = mxfp4._MXFP4Forward.forward(input, weight, True, False, False, mode)[0]

    torch.testing.assert_close(output, input @ weight.T)
    if mode == 3:
        assert used_weights == [forward_weight, forward_weight]


def test_mxfp4_capture_is_rank_zero_and_deduplicated(monkeypatch, tmp_path):
    mxfp4 = _load_mxfp4(monkeypatch)
    monkeypatch.setenv("FLUX_MXFP4_CAPTURE_STEPS", "7")
    monkeypatch.setenv("FLUX_MXFP4_CAPTURE_DIR", str(tmp_path))
    mxfp4._MXFP4_CAPTURED.clear()
    mxfp4.set_mxfp4_capture_step(7)
    input = torch.arange(12).reshape(3, 4)
    weight = torch.ones(2, 4)

    mxfp4._save_mxfp4_capture(input, weight, "double_blocks.0.img_mlp.0")
    mxfp4._save_mxfp4_capture(input + 1, weight, "double_blocks.0.img_mlp.0")

    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1
    capture = torch.load(files[0], weights_only=True)
    assert capture["step"] == 7
    assert capture["module"] == "double_blocks.0.img_mlp.0"
    torch.testing.assert_close(capture["input"], input)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    mxfp4._save_mxfp4_capture(input, weight, "double_blocks.1.img_mlp.0")
    assert list(tmp_path.glob("*.pt")) == files


def test_mxfp4_trainer_heals_resumed_run_after_threshold_and_resets_dynamo(monkeypatch):
    from omniflow.trainers.base import BaseWanTrainer

    class HealOnce(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def switch_forward_to_bf16(self):
            self.calls += 1
            return self.calls == 1

    trainer = BaseWanTrainer.__new__(BaseWanTrainer)
    trainer.model = HealOnce()
    trainer.rank = 0
    resets = []
    monkeypatch.setenv("FLUX_MXFP4_BF16_FORWARD_SWITCH_STEP", "7")
    monkeypatch.setattr(torch._dynamo, "reset", lambda: resets.append(True))

    trainer.global_step = 6
    trainer._prepare_mxfp4_forward()
    assert trainer.model.calls == 0
    trainer.global_step = 9
    trainer._prepare_mxfp4_forward()
    trainer._prepare_mxfp4_forward()
    assert trainer.model.calls == 1
    assert resets == [True]


def test_mxfp4_forward_healing_is_one_time_and_keeps_backward_precision(monkeypatch):
    mxfp4 = _load_mxfp4(monkeypatch)
    module = mxfp4.MXFP4Linear(
        torch.nn.Linear(4, 3),
        "mxfp4",
        forward_hadamard=True,
        activation_residual_mode=3,
    )

    assert not hasattr(module, "_function")
    assert module.switch_forward_to_bf16() is True
    assert module.switch_forward_to_bf16() is False
    assert module.forward_precision == "bf16"
    assert module.backward_precision == "mxfp4"
    assert module.forward_hadamard is False
    assert module.activation_residual_mode == 0


def test_mxfp4_eval_bf16_bypasses_strategy_and_gradient_sr_reaches_training(monkeypatch):
    mxfp4 = _load_mxfp4(monkeypatch)
    calls = []

    class FakeMXFP4Forward:
        @staticmethod
        def apply(*args):
            calls.append(args)
            return (torch.nn.functional.linear(args[0], args[1]),)

    monkeypatch.setitem(mxfp4._FORWARD_FUNCTIONS, "mxfp4", FakeMXFP4Forward)
    linear = torch.nn.Linear(4, 3)
    module = mxfp4.MXFP4Linear(
        linear,
        "mxfp4",
        eval_bf16=True,
        gradient_stochastic_rounding=True,
    )
    input = torch.randn(2, 4)

    module.train()
    module(input)
    assert calls[0][3] is True

    module.eval()
    calls.clear()
    torch.testing.assert_close(module(input), torch.nn.functional.linear(input, linear.weight, linear.bias))
    assert calls == []


@pytest.mark.parametrize("profile", ["pareto_a", "pareto_b", "custom"])
def test_mxfp4_profiles_use_mixed_quant_image(profile):
    env = os.environ.copy()
    env.pop("DOCKER_IMAGE", None)
    command = (
        f"source examples/mlperf/flux1/config_4n_gbs1024_mxfp4_{profile}.sh; "
        'printf "%s" "$DOCKER_IMAGE"'
    )
    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True, env=env
    )
    assert result.stdout == "zirui3/primus-v26.3-flux:v0.4-mxfp4-mixed-quant-uos"


@pytest.mark.parametrize("profile", ["pareto_a", "pareto_b"])
def test_mxfp4_pareto_profiles_use_bf16_evaluation(profile):
    command = (
        f"source examples/mlperf/flux1/config_4n_gbs1024_mxfp4_{profile}.sh; "
        'printf "%s" "$FLUX_MXFP4_EVAL_PRECISION"'
    )
    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )
    assert result.stdout == "bf16"


def test_mxfp4_capture_dir_is_defaulted_after_output_dir(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env bash\nprintf "%s" "$FLUX_MXFP4_CAPTURE_DIR"\n')
    docker.chmod(0o755)
    output_dir = tmp_path / "inside-output"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DATA_ROOT": str(tmp_path),
        "OUTPUT_ROOT": str(tmp_path / "host-output"),
        "OUTPUT_DIR": str(output_dir),
        "FLUX_CONFIG": "config_4n_gbs1024_mxfp4_custom.sh",
        "NCCL_IB_DISABLE": "1",
    }
    env.pop("FLUX_MXFP4_CAPTURE_DIR", None)
    result = subprocess.run(
        ["bash", "examples/mlperf/flux1/run_with_docker.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.endswith(f"{output_dir}/mxfp4-captures")


def test_mxfp4_custom_profile_preserves_environment_overrides():
    overrides = {
        "FLUX_MXFP4_FORWARD_PRECISION": "bf16",
        "FLUX_MXFP4_BF16_FORWARD_SCOPE": "double_all",
        "FLUX_MXFP4_SELECTIVE_FORWARD_SCOPE": "single_linear2",
        "FLUX_MXFP4_FORWARD_HADAMARD": "mlp",
        "FLUX_MXFP4_ACTIVATION_RESIDUAL": "all",
        "FLUX_MXFP4_ACTIVATION_RESIDUAL_DTYPE": "bf16",
        "FLUX_MXFP4_EVAL_PRECISION": "bf16",
        "FLUX_MXFP4_GRADIENT_SR": "true",
        "PRIMUS_TURBO_GEMM_BACKEND": "FP4:TEST",
        "PRIMUS_TURBO_AUTO_TUNE": "1",
    }
    names = list(overrides)
    command = (
        "source examples/mlperf/flux1/config_4n_gbs1024_mxfp4_custom.sh; "
        + "printf '%s\\n' "
        + " ".join(f'"${{{name}}}"' for name in names)
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **overrides},
    )
    assert result.stdout.splitlines() == list(overrides.values())
