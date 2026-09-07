#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_dp8_fp8_allgather.sh"

export NNODES=1
export DP_REPLICATE=1
export GLOBAL_BATCH_SIZE=256
export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
