
# Set cluster ENV
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Set debug ENV
export FIXED_TIMESTEP=500
export FIXED_SEED=10007
export ALIGN_WITH_DIFFSYNTH="1"
# export WAN_DEBUG=1
export FP32_MASTER_WEIGHTS='1'

export TOKENIZERS_PARALLELISM=false


# wan2.2-5B
CONFIG="examples/debug_fsdp2_new2_5b_sp.yml"

# Launch training
torchrun \
    --nnodes=${NNODES} --node_rank=${NODE_RANK} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config ${CONFIG}