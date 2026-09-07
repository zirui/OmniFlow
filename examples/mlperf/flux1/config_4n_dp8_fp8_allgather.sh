#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024.sh"

export FLUX_FP8_ALL_GATHER=1
