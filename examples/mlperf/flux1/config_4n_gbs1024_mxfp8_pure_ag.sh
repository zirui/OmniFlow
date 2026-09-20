#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024_mxfp8_pure.sh"

export FLUX_MXFP8_ALL_GATHER=1
export CUDA_DEVICE_MAX_CONNECTIONS=4
export GPU_MAX_HW_QUEUES=4
