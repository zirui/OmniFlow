#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024_mxfp4_pareto_b.sh"

export PRIMUS_TURBO_GEMM_BACKEND=FP4:FLYDSL
export PRIMUS_TURBO_AUTO_TUNE=1
export FLUX_MXFP4_FLYDSL_QUANT=1
