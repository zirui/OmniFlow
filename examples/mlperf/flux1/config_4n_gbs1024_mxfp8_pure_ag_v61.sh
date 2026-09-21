#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024_mxfp8_pure_ag.sh"

export EVAL_BATCH_SIZE=116
export LR=0.000275
export WARMUP_STEPS=800
