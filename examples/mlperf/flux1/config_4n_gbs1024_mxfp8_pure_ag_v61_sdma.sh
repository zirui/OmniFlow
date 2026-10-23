#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024_mxfp8_pure_ag_v61.sh"

export FSDP_ALL_GATHER_BACKEND=rccl_sdma
export FSDP_SDMA_MODULES=${FSDP_SDMA_MODULES:-single_blocks}
export FLUX_SDMA_ATTACH=1
export NCCL_CTA_POLICY=2
export NCCL_CUMEM_ENABLE=1
export NCCL_LOCAL_REGISTER=0
export NCCL_SYM_GIN_KERNELS_ENABLE=0
export NCCL_HIER_CE_COLL_NUM_CTX=1
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
