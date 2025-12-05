#!/bin/bash

CONFIG="configs/wan2.2_t2v_5b.yml"

# Number of GPUs
NGPUS=1

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12356 \
  train.py --config ${CONFIG}

