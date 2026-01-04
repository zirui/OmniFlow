#!/bin/bash

CURDIR=$(cd $(dirname $0); pwd)

# CONFIG="configs/wan2.2_t2v_5b.yml"
CONFIG="configs/wan2.1_t2v_1.3b_sft.yml"

# Number of GPUs
NGPUS=1

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/workspace/common_module:$CURDIR/../
export WANDB_DISABLED=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12356 \
  train.py --config ${CONFIG} \

