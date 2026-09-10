#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the MLPerf datasets}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the output directory}"

export FLUX_CONFIG=${FLUX_CONFIG:-config_4n_gbs1024.sh}
[[ -f "$SCRIPT_DIR/$FLUX_CONFIG" ]] || { echo "Unknown FLUX_CONFIG: $FLUX_CONFIG" >&2; exit 2; }
source "$SCRIPT_DIR/$FLUX_CONFIG"
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$LOCAL_BATCH_SIZE}

export DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4}
export CONTAINER_NAME=${CONTAINER_NAME:-omniflow-mlperf-flux1-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export CONFIG=${CONFIG:-examples/mlperf/flux1/flux.1_schnell_t2i-native.yaml}
export DATASET_PATH=${DATASET_PATH:-/data/cc12m_preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export OUTPUT_DIR=${OUTPUT_DIR:-/output/flux_mlperf}
export FLUX_MXFP4_CAPTURE_DIR=${FLUX_MXFP4_CAPTURE_DIR:-$OUTPUT_DIR/mxfp4-captures}
export MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-$OUTPUT_DIR/mlperf_compliance.log}
export PROFILE=${PROFILE:-false}
export PROFILE_RANK=${PROFILE_RANK:-0}
export PROFILE_WAIT_STEPS=${PROFILE_WAIT_STEPS:-30}
export PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS:-2}
export PROFILE_ACTIVE_STEPS=${PROFILE_ACTIVE_STEPS:-5}
export PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-$OUTPUT_DIR/torch_profile}
export PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}

actual_gbs=$((NNODES * GPUS_PER_NODE * LOCAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
[[ "$actual_gbs" == "$GLOBAL_BATCH_SIZE" ]] || {
    echo "Invalid $FLUX_CONFIG: expected GBS=$GLOBAL_BATCH_SIZE, got $actual_gbs" >&2
    exit 2
}
printf '%s\n' "[flux1] config=$FLUX_CONFIG nnodes=$NNODES mbs=$LOCAL_BATCH_SIZE "\
    "ga=$GRADIENT_ACCUMULATION_STEPS gbs=$actual_gbs eval_bs=$EVAL_BATCH_SIZE "\
    "eval_workers=$EVAL_DATALOADER_NUM_WORKERS eval_prefetch=$EVAL_DATALOADER_PREFETCH_FACTOR"

if (( NNODES > 1 )) && [[ "${NCCL_IB_DISABLE:-0}" != "1" ]]; then
    LIBIONIC_ABI4_PATH=${LIBIONIC_ABI4_PATH:-/usr/lib/x86_64-linux-gnu/libionic.so}
    [[ -f "$LIBIONIC_ABI4_PATH" && -e /dev/infiniband ]] || {
        echo "ABI-4 libionic or /dev/infiniband is unavailable" >&2
        exit 1
    }
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=${NCCL_IB_HCA:-ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1}
    export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-1}
    export NCCL_IB_TC=${NCCL_IB_TC:-104}
    export NCCL_IB_FIFO_TC=${NCCL_IB_FIFO_TC:-192}
    export NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM:-2}
    export NCCL_IB_USE_INLINE=${NCCL_IB_USE_INLINE:-1}
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}
    export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-20}
    export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-300}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens3}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens3}
    export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-librccl-anp.so}
    export NCCL_MAX_P2P_CHANNELS=${NCCL_MAX_P2P_CHANNELS:-56}
    export NCCL_GDR_FLUSH_DISABLE=${NCCL_GDR_FLUSH_DISABLE:-1}
    export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-0}
    export NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY:-1}
    export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
    export NET_OPTIONAL_RECV_COMPLETION=${NET_OPTIONAL_RECV_COMPLETION:-1}
    export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=${RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING:-0}
fi

env_names=(
    FLUX_CONFIG NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE HIP_VISIBLE_DEVICES DP_REPLICATE CONFIG
    DATASET_PATH EVAL_DATASET_PATH EMPTY_ENCODINGS_PATH OUTPUT_DIR
    MLLOG_OUTPUT_FILE MLLOG_SUBMISSION_DIVISION MLLOG_SUBMISSION_ORG MLLOG_SUBMISSION_PLATFORM
    MLLOG_SUBMISSION_POC_NAME MLLOG_SUBMISSION_POC_EMAIL MLLOG_SUBMISSION_STATUS
    FLUX_FLOAT8_RECIPE FLUX_FP8_GEMM_BACKEND FLUX_MXFP4_RECIPE
    FLUX_MXFP4_FORWARD_PRECISION FLUX_MXFP4_FORWARD_HADAMARD
    FLUX_MXFP4_BF16_FORWARD_SCOPE FLUX_MXFP4_SELECTIVE_FORWARD_SCOPE
    FLUX_MXFP4_ACTIVATION_RESIDUAL FLUX_MXFP4_ACTIVATION_RESIDUAL_DTYPE
    FLUX_MXFP4_EVAL_PRECISION FLUX_MXFP4_GRADIENT_SR
    FLUX_MXFP4_CAPTURE_STEPS FLUX_MXFP4_CAPTURE_MODULES FLUX_MXFP4_CAPTURE_DIR
    FLUX_MXFP4_BF16_FORWARD_SWITCH_STEP ATTENTION_BACKEND
    LOCAL_BATCH_SIZE EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS GLOBAL_BATCH_SIZE MAX_STEPS LR WARMUP_STEPS
    DATALOADER_NUM_WORKERS EVAL_DATALOADER_NUM_WORKERS EVAL_DATALOADER_PREFETCH_FACTOR
    GRADIENT_CHECKPOINTING GRADIENT_CHECKPOINTING_RATIO COMPILE_TRANSFORMER_BLOCKS COMPILE_STRATEGY
    COMPILE_BACKEND COMPILE_FULLGRAPH COMPILE_DYNAMIC COMPILE_OUTPUT_HEAD TORCH_COMPILE_MODE
    TORCHINDUCTOR_CACHE_DIR TORCHINDUCTOR_CACHE_SEED TORCHINDUCTOR_CACHE_EXPORT
    FSDP2_RESHARD_AFTER_FORWARD FSDP2_REDUCE_DTYPE FLUX_FP8_ALL_GATHER
    PROFILE PROFILE_RANK PROFILE_WAIT_STEPS PROFILE_WARMUP_STEPS PROFILE_ACTIVE_STEPS
    PROFILE_OUTPUT_DIR PROFILE_WITH_STACK FSDP2_HSDP_FP8_ALL_REDUCE FSDP2_HSDP_FP8_BLOCK_SIZE
    FSDP2_HSDP_FP8_MIN_NUMEL PIN_FLUX_T5_STACK
    FLUX_PERFORMANCE_MODE SAVE_STEPS SAVE_STRATEGY CHECKPOINT_KEEP_LATEST
    RESUME_FROM_CHECKPOINT MLPERF_ENABLE MLPERF_WARMUP_TRAIN_STEPS
    MLPERF_WARMUP_VALIDATION_STEPS MLPERF_CLEAR_CACHES TARGET_ACCURACY
    VAL_CHECK_INTERVAL MLPERF_VALIDATION_START_STEP SEED LOG_FREQ ENABLE_WANDB_LOGGER
    TORCHINDUCTOR_BENCHMARK_FUSION
    PRIMUS_FLUX_AITER_ATOMIC_FP32 PRIMUS_FLUX_REUSE_FP8_INPUT
    PRIMUS_TURBO_GEMM_BACKEND PRIMUS_TURBO_AUTO_TUNE NCCL_IB_DISABLE
    NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_TC NCCL_IB_FIFO_TC NCCL_IB_ROCE_VERSION_NUM
    NCCL_IB_USE_INLINE NCCL_IB_QPS_PER_CONNECTION NCCL_IB_RETRY_CNT NCCL_IB_TIMEOUT
    NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET_PLUGIN NCCL_MAX_P2P_CHANNELS
    NCCL_GDR_FLUSH_DISABLE NCCL_DMABUF_ENABLE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
    NCCL_IGNORE_CPU_AFFINITY NCCL_CROSS_NIC NCCL_DEBUG NCCL_DEBUG_SUBSYS
    NET_OPTIONAL_RECV_COMPLETION RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING
    HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM GPU_MAX_HW_QUEUES CUDA_DEVICE_MAX_CONNECTIONS
    TORCH_NCCL_HIGH_PRIORITY NCCL_CHECKS_DISABLE NCCL_PXN_DISABLE NCCL_P2P_NET_CHUNKSIZE
    TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK
)
docker_env_args=()
for name in "${env_names[@]}"; do
    [[ -v "$name" ]] && docker_env_args+=(--env "$name")
done

rdma_args=()
libionic_args=()
[[ -e /dev/infiniband ]] && rdma_args=(--device=/dev/infiniband)
[[ -n "${LIBIONIC_ABI4_PATH:-}" ]] && libionic_args=(-v "$LIBIONIC_ABI4_PATH:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187:ro")
mkdir -p "$OUTPUT_ROOT"

exec docker run --rm --init --privileged \
    --name "$CONTAINER_NAME" \
    --ulimit nofile=1048576:1048576 --ulimit memlock=-1:-1 \
    --device=/dev/kfd --device=/dev/dri "${rdma_args[@]}" --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --network=host --shm-size=20G \
    -v "$REPO_ROOT:/workspace/OmniFlow" \
    -v "$DATA_ROOT:/data" \
    -v "$OUTPUT_ROOT:/output" \
    -v /shared_nfs:/shared_nfs \
    "${libionic_args[@]}" \
    -w /workspace/OmniFlow \
    "${docker_env_args[@]}" \
    "$DOCKER_IMAGE" bash -lc '
        set -euo pipefail
        mkdir -p "$OUTPUT_DIR"
        if [[ -n "${TORCHINDUCTOR_CACHE_SEED:-}" ]]; then
            export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache
            cache_seed=${TORCHINDUCTOR_CACHE_SEED//%r/$NODE_RANK}
            rm -rf "$TORCHINDUCTOR_CACHE_DIR"
            mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
            if [[ -d "$cache_seed" ]]; then
                cp -a "$cache_seed/." "$TORCHINDUCTOR_CACHE_DIR/"
            elif [[ -f "$cache_seed" ]]; then
                tar --zstd -xf "$cache_seed" -C "$TORCHINDUCTOR_CACHE_DIR"
            else
                echo "Missing Inductor cache seed: $cache_seed" >&2
                exit 1
            fi
            echo "[flux1] loaded Inductor cache seed $cache_seed into $TORCHINDUCTOR_CACHE_DIR"
        fi
        if [[ "$MLPERF_CLEAR_CACHES" == "true" ]]; then
            sync
            echo 3 > /proc/sys/vm/drop_caches
        fi
        torchrun \
          --nnodes="$NNODES" \
          --node_rank="$NODE_RANK" \
          --nproc_per_node="$GPUS_PER_NODE" \
          --master_addr="$MASTER_ADDR" \
          --master_port="$MASTER_PORT" \
          train.py --config "$CONFIG"
        if [[ -n "${TORCHINDUCTOR_CACHE_EXPORT:-}" && ( "$NODE_RANK" == "0" || "$TORCHINDUCTOR_CACHE_EXPORT" == *%r* ) ]]; then
            [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]] || {
                echo "TORCHINDUCTOR_CACHE_EXPORT requires a cache directory or seed" >&2
                exit 1
            }
            cache_export=${TORCHINDUCTOR_CACHE_EXPORT//%r/$NODE_RANK}
            tmp_export="$cache_export.tmp.$$"
            rm -f "$tmp_export"
            tar --zstd -cf "$tmp_export" -C "$TORCHINDUCTOR_CACHE_DIR" .
            mv "$tmp_export" "$cache_export"
            echo "[flux1] exported Inductor cache to $cache_export"
        fi
    '
