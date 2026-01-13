#!/bin/bash

CONFIG="configs/wan2.1_t2v_1.3b_sft.yml" 

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
NNODES=${NNODES:-1}
NGPUS=${NGPUS:-1}
MASTER_PORT=${MASTER_PORT:-12356}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

export PYTHONPATH=/zirui/code/OmniFlow/src

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=${NNODES} \
  --node_rank=0 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py --config ${CONFIG} \

