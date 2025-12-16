#!/bin/bash
set -e

# Set PYTHONPATH to include the project root
ROOT_DIR=/root/zirui/code/OmniFlow
export PYTHONPATH=${ROOT_DIR}/src/omniflow:${ROOT_DIR}/src:$ROOT_DIR:${PWD}:$PYTHONPATH
echo $ROOT_DIR

# Distributed args
NNODES=1
NPROC_PER_NODE=1  # Set to 8 for full node, 1 for debug
MASTER_ADDR="localhost"
MASTER_PORT="29500"

CONFIG_FILE="torchtitan/experiments/wan/train_configs/wan2.1_t2v_debug.toml"

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "Starting training with config: $CONFIG_FILE"


# Launch the training script
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    --local_ranks_filter=0 --role=rank --tee=3 \
    torchtitan/experiments/wan/train.py \
    --job.config_file ${CONFIG_FILE}
