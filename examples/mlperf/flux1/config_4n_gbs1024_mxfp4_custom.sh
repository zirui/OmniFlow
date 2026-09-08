#!/usr/bin/env bash

_float8_recipe=${FLUX_FLOAT8_RECIPE-}
_fp8_backend=${FLUX_FP8_GEMM_BACKEND-}
_mxfp4_recipe=${FLUX_MXFP4_RECIPE:-custom}
_forward_precision=${FLUX_MXFP4_FORWARD_PRECISION:-mxfp8}
_bf16_scope=${FLUX_MXFP4_BF16_FORWARD_SCOPE:-double_img_up_2_3}
_selective_scope=${FLUX_MXFP4_SELECTIVE_FORWARD_SCOPE:-late_txt_mlp_up}
_forward_hadamard=${FLUX_MXFP4_FORWARD_HADAMARD:-all}
_activation_residual=${FLUX_MXFP4_ACTIVATION_RESIDUAL:-double_txt_up}
_residual_dtype=${FLUX_MXFP4_ACTIVATION_RESIDUAL_DTYPE:-mxfp8}
_eval_precision=${FLUX_MXFP4_EVAL_PRECISION:-same}
_gradient_sr=${FLUX_MXFP4_GRADIENT_SR:-false}
_fp4_backend=${PRIMUS_TURBO_GEMM_BACKEND:-FP4:AITER}
_auto_tune=${PRIMUS_TURBO_AUTO_TUNE:-0}

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024.sh"

export DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4-mxfp4-mixed-quant-uos}
export FLUX_FLOAT8_RECIPE=$_float8_recipe
export FLUX_FP8_GEMM_BACKEND=$_fp8_backend
export FLUX_MXFP4_RECIPE=$_mxfp4_recipe
export FLUX_FP8_ALL_GATHER=0
export PRIMUS_FLUX_REUSE_FP8_INPUT=0
export FLUX_MXFP4_FORWARD_PRECISION=$_forward_precision
export FLUX_MXFP4_BF16_FORWARD_SCOPE=$_bf16_scope
export FLUX_MXFP4_SELECTIVE_FORWARD_SCOPE=$_selective_scope
export FLUX_MXFP4_FORWARD_HADAMARD=$_forward_hadamard
export FLUX_MXFP4_ACTIVATION_RESIDUAL=$_activation_residual
export FLUX_MXFP4_ACTIVATION_RESIDUAL_DTYPE=$_residual_dtype
export FLUX_MXFP4_EVAL_PRECISION=$_eval_precision
export FLUX_MXFP4_GRADIENT_SR=$_gradient_sr
export PRIMUS_TURBO_GEMM_BACKEND=$_fp4_backend
export PRIMUS_TURBO_AUTO_TUNE=$_auto_tune
