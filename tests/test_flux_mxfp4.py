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
