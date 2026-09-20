#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024_mxfp8_pure_ag.sh"

export NNODES=2
export DP_REPLICATE=2
export GRADIENT_ACCUMULATION_STEPS=2
