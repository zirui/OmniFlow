#!/bin/bash
#SBATCH --job-name=omniflow
#SBATCH --output=logs/omniflow.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00

##SBATCH --partition=xxx
##SBATCH --exclude=xxx

# ---- User Config ----
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v25.9_gfx942"}
CONFIG=${CONFIG:-"examples/debug_fsdp2_new2_5b_sp.yml"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ---- Distributed Setup ----
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"
echo "NNODES=$SLURM_JOB_NUM_NODES, GPUS_PER_NODE=$GPUS_PER_NODE"

# ---- Pull image ----
srun bash -c 'docker pull "$DOCKER_IMAGE"'

# ---- Launch ----
srun bash -c "\
docker ps -aq --filter name=t2v-train | xargs -r docker rm -f
docker run --rm \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video --privileged \
    -v /etc/libibverbs.d/:/etc/libibverbs.d \
    -v /mnt/shared/zirui/:/mnt/shared/zirui \
    -v /mnt/shared/zirui/:/zirui \
    -w /zirui/code/OmniFlow \
    --name t2v-train-\$SLURM_NODEID \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e NNODES=\$SLURM_JOB_NUM_NODES \
    -e NODE_RANK=\$SLURM_NODEID \
    -e WANDB_API_KEY=${WANDB_API_KEY:-73ba028fade29b4baafdba2d6996a4865e28f410} \
    -e WANDB_PROJECT=${WANDB_PROJECT:-t2v-debug} \
    -e TOKENIZERS_PARALLELISM=false \
    -e FIXED_TIMESTEP=500 \
    -e FIXED_SEED=10007 \
    -e ALIGN_WITH_DIFFSYNTH="1" \
    -e FP32_MASTER_WEIGHTS='1' \
    -e PYTHONPATH=/zirui/code/OmniFlow/src \
    $DOCKER_IMAGE bash -c '\
        # install binutils
        apt-get update -y && \
        apt-get install -y binutils && \
        # Activate env
        source /zirui/mm-env12/bin/activate && \
        torchrun \
            --nnodes=\$NNODES \
            --node_rank=\$NODE_RANK \
            --nproc_per_node=$GPUS_PER_NODE \
            --master_addr=\$MASTER_ADDR \
            --master_port=\$MASTER_PORT \
            train.py \
            --config $CONFIG
    '
"
