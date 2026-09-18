#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_1n_gbs1024.sh"

export FLUX_FP8_QKV_WGRAD_HIGH_PRECISION=false
