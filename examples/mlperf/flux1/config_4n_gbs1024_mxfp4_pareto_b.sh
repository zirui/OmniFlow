#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024.sh"

export DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4-mxfp4-uos}
export FLUX_FLOAT8_RECIPE=
export FLUX_FP8_GEMM_BACKEND=
export DP_REPLICATE=4
export FLUX_MXFP4_RECIPE=pareto_b
export FLUX_MXFP4_EVAL_PRECISION=bf16
export FLUX_FP8_ALL_GATHER=0
export PRIMUS_FLUX_REUSE_FP8_INPUT=0
export PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER
export PRIMUS_TURBO_AUTO_TUNE=0
