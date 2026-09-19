#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_1n_gbs1024_mxfp8_pure.sh"

export FLUX_MXFP8_ALL_GATHER=1
export COMPILE_TRANSFORMER_BLOCKS=${MXFP8_AG_COMPILE_TRANSFORMER_BLOCKS:-true}
