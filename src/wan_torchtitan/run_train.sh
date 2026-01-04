#!/bin/bash
set -e

# Set PYTHONPATH to include the project root
# ROOT_DIR=/root/zirui/code/OmniFlow
# export PYTHONPATH=${ROOT_DIR}/src/omniflow:${ROOT_DIR}/src:$ROOT_DIR:${PWD}:$PYTHONPATH
# echo $ROOT_DIR
CURDIR=$(cd $(dirname $0); pwd)
export PYTHONPATH=/workspace/common_module/:`pwd`/../../src/:`pwd`/../../src/omniflow/

# Distributed args
NNODES=1
NPROC_PER_NODE=1  # Set to 8 for full node, 1 for debug
MASTER_ADDR="localhost"
MASTER_PORT="23500"

CONFIG_FILE="torchtitan/experiments/wan/train_configs/wan2.1_t2v_1.3b_sft.toml"

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "Starting training with config: $CONFIG_FILE"


# Launch the training script
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    --local_ranks_filter=0 --role=rank --tee=3 \
    -m torchtitan.experiments.wan.train \
    --job.config_file ${CONFIG_FILE}
