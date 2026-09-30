# FLUX multi-node development notes

Start from the OmniFlow repository root:

```bash
REPO=/shared_nfs/zirui/code/OmniFlow
DATA_ROOT=/shared_nfs/zirui/data
OUTPUT_ROOT=/shared_nfs/zirui/runs/flux-$(date -u +%Y%m%dT%H%M%SZ)
DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4.3
cd "$REPO"
export DOCKER_IMAGE
```

The only launch path is native PyTorch:

```text
run_with_docker_slurm.sh
  -> one run_with_docker.sh process per node
  -> one container per node
  -> torchrun --nproc-per-node=8 train.py --config ...
```

Do not start one Slurm task per GPU. `torchrun` owns the node-local workers.

## One allocation

```bash
export DATA_ROOT OUTPUT_ROOT
FLUX_CONFIG=config_2n_gbs1024.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

### Single-node MXFP4 performance settings

For GBS/MBS/GA `1024/32/4` on one Crusoe MI355X node, use native FP8
AllGather, an exact recipe-specific Inductor cache, and the qualified scheduling
defaults below:

```bash
export DP_REPLICATE=1 FLUX_FP8_ALL_GATHER=1 FSDP2_FORCE_SUM=1
export GPU_MAX_HW_QUEUES=2 HSA_NO_SCRATCH_RECLAIM=1
unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_NTHREADS NCCL_ALGO NCCL_PROTO
unset FLUX_NCCL_NET_PLUGIN FLUX_NCCL_DMABUF_ENABLE
unset NCCL_NET_PLUGIN NCCL_DMABUF_ENABLE
```

Let the profile restore the Crusoe RCCL-ANP and DMA-BUF defaults after the
`unset` commands. Do not reuse DCCS channel or plugin overrides on one node.
Matched short runs found one recipe-specific difference:

- 0-BF16 FRQ19: leave `CUDA_DEVICE_MAX_CONNECTIONS` unset.
- Pareto-B: set `CUDA_DEVICE_MAX_CONNECTIONS=1`.

Both recipes regress with `GPU_MAX_HW_QUEUES=4` and
`CUDA_DEVICE_MAX_CONNECTIONS=5`. Forced 48 channels are slightly slower than
the settings above, while forced 24 channels substantially regress 0-BF16.

### Four-node performance settings

DCCS is the performance qualification platform. Keep its DP32 recipe and site
settings unchanged:

```bash
DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4.1 \
FLUX_CONFIG=config_4n_gbs1024.sh \
DP_REPLICATE=1 FLUX_FP8_ALL_GATHER=1 \
TORCH_COMPILE_MODE=default LOG_FREQ=1 \
NCCL_DMABUF_ENABLE=0 \
NCCL_SOCKET_IFNAME=fenic GLOO_SOCKET_IFNAME=fenic \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

Crusoe has a host-network AllGather limitation. Its best validated workaround
keeps each FSDP shard group node-local with HSDP 4x8 and uses exact per-rank
Inductor caches:

```bash
OUTPUT_ROOT=/path/containing/cache-node0..3.tar.zst \
DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4.1 \
FLUX_CONFIG=config_4n_gbs1024.sh DP_REPLICATE=4 FLUX_FP8_ALL_GATHER=1 \
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
TORCHINDUCTOR_CACHE_SEED=/output/cache-node%r.tar.zst \
HSA_NO_SCRATCH_RECLAIM=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_DMABUF_ENABLE=1 NCCL_SOCKET_IFNAME=ens3 GLOO_SOCKET_IFNAME=ens3 \
NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1 NCCL_CROSS_NIC=0 \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

Do not apply the Crusoe overrides on DCCS. Keep gradient reduction in BF16;
FP8 HSDP AllReduce changes numerics and is not a qualified performance result.
Rebuild the exact caches after changing the image, graph, shapes, or compile
settings.

The allocation node count must match the selected profile. For a short smoke
test, add:

```bash
MAX_STEPS=1 SAVE_STRATEGY=none MLPERF_CLEAR_CACHES=false
```

## Separate single-node allocations

A single `srun` cannot span multiple job IDs. Start one launcher in each
allocation with the same `NNODES`, `MASTER_ADDR`, and `MASTER_PORT`, and assign
contiguous `NODE_RANK` values:

```bash
spur run --jobid=<job-id> --overlap -N1 -n1 --nodelist=<node> \
  env DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
      NNODES=2 NODE_RANK=<rank> MASTER_ADDR=<rank-0-ip> MASTER_PORT=29601 \
      FLUX_CONFIG=config_2n_gbs1024.sh \
      bash "$REPO/examples/mlperf/flux1/run_with_docker.sh"
```

Every node must see the same repository, datasets, output path, and launch
environment. The same pattern also works over direct SSH: start all commands
concurrently, replace `spur run ... env` with `ssh <node> env`, and keep the
same rendezvous values plus one unique contiguous `NODE_RANK` per node. Use
only nodes owned by the current allocation.

## Direct SSH across four nodes (DCCS / Crusoe)

Use this path when the four nodes are already available but are not in one
scheduler allocation. Run one `run_with_docker.sh` over SSH per node; do not
try to span the nodes with one `srun`. Use only nodes allocated to you or
explicitly shared by their owner.

Set `REPO`, `DATA_ROOT`, and `OUTPUT_ROOT` to shared `/perf_apps` paths on DCCS
or shared `/shared_nfs` paths on Crusoe, then edit only the node list:

```bash
nodes=(node0 node1 node2 node3)
MASTER_ADDR=$(getent ahostsv4 "${nodes[0]}" | awk 'NR == 1 {print $1}')
MASTER_PORT=${MASTER_PORT:-29601}
mkdir -p "$OUTPUT_ROOT/ssh-launch"

SITE_ENV= # Crusoe defaults
if [[ ${nodes[0]} == smci355-* ]]; then
  SITE_ENV='NCCL_DMABUF_ENABLE=0 NCCL_SOCKET_IFNAME=fenic GLOO_SOCKET_IFNAME=fenic'
fi

pids=()
for rank in "${!nodes[@]}"; do
  node=${nodes[$rank]}
  ssh "$node" "cd '$REPO' && env $SITE_ENV \
    DATA_ROOT='$DATA_ROOT' OUTPUT_ROOT='$OUTPUT_ROOT' \
    FLUX_CONFIG=config_4n_gbs1024.sh NNODES=4 NODE_RANK=$rank \
    MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' \
    CONTAINER_NAME=omniflow-flux-4n-$rank \
    bash examples/mlperf/flux1/run_with_docker.sh" \
    >"$OUTPUT_ROOT/ssh-launch/$node.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
```

Add recipe overrides (image, FP8/MXFP4 settings, cache, smoke-test limits, and
so on) beside `FLUX_CONFIG` so every rank receives the same values. Choose a
fresh `MASTER_PORT` and unique container-name prefix for concurrent runs.

On Crusoe, first configure job-scoped SSH for every owning allocation with
`scripts/spur-ssh --setup-only JOB_ID` from the xutils checkout; then verify
`ssh NODE hostname` for all four nodes. On DCCS, direct compute-node SSH is
intended only after the nodes have been allocated or explicitly shared.

Monitor with `tail -f "$OUTPUT_ROOT/ssh-launch/"*.log`. To stop a run, kill
only the four `omniflow-flux-4n-*` containers; do not cancel a shared
allocation.

## Inductor cache

Do not let all ranks populate one writable NFS cache. Prewarm one GPU, then use
the exported archive for distributed runs:

```bash
CACHE_DIR=/shared_nfs/zirui/runs/flux-inductor-seed
mkdir -p "$CACHE_DIR"

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=20G \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/workspace/OmniFlow/src \
  -e TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR" \
  -v "$REPO:/workspace/OmniFlow" -v /shared_nfs:/shared_nfs \
  -w /workspace/OmniFlow "$DOCKER_IMAGE" \
  python examples/mlperf/flux1/prewarm_inductor_cache.py

# Keep graph choices but force the exact distributed build to materialize its
# own Triton artifacts.
GENERIC_CACHE="$CACHE_DIR-generic.tar.zst"
tar --exclude=./triton --zstd -cf "$GENERIC_CACHE" -C "$CACHE_DIR" .

TORCHINDUCTOR_CACHE_SEED="$GENERIC_CACHE" \
TORCHINDUCTOR_CACHE_EXPORT="$CACHE_DIR-exact.tar.zst" \
MAX_STEPS=1 MLPERF_ENABLE=false SAVE_STRATEGY=none \
DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

For native FP8 AllGather, build an exact cache with one independent 8-GPU run
per cache-producing node, then route node-local archives with `%r`:

```bash
# Run separately on cache-producing nodes; use a distinct OUTPUT_ROOT each time.
TORCHINDUCTOR_CACHE_SEED="$GENERIC_CACHE" \
TORCHINDUCTOR_CACHE_EXPORT=/output/cache-node0.tar.zst \
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
FLUX_FP8_ALL_GATHER=1 MAX_STEPS=20 MLPERF_ENABLE=true SAVE_STRATEGY=none \
DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
FLUX_CONFIG=config_1n_gbs1024.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh

# Place cache-node0.tar.zst ... cache-node3.tar.zst under OUTPUT_ROOT.
TORCHINDUCTOR_CACHE_SEED=/output/cache-node%r.tar.zst \
FLUX_FP8_ALL_GATHER=1 \
DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
FLUX_CONFIG=config_4n_gbs1024.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

On homogeneous MI355X nodes, a verified archive may be copied to another node
rank. Never let multiple ranks write one shared cache. Rebuild the cache when
the image, PyTorch/Triton/FlyDSL versions, model graph, batch shapes, or compile
settings change.

## Multi-node caveats

- DCCS is the final performance reference. Matched tests show no equivalent
  host-network regression there; do not use Crusoe-only workarounds to tune or
  judge DCCS performance.
- Crusoe's 2--8 MiB cross-node AllGather reaches only 61--63 GB/s/rank versus
  195--232 GB/s/rank on DCCS. The evidence points to its Ionic driver and
  VF/PCI topology, not the training code or `/shared_nfs`.
- On Crusoe, keep `NCCL_CROSS_NIC=0`; setting it to 1 causes Ionic CQE errors.
  Four QPs, forced 48 channels, and `RCCL_WARP_SPEED_AUTO=0` did not help.
- A single `srun` cannot span job IDs. Launch all ranks concurrently with one
  rendezvous and unique contiguous `NODE_RANK` values.
- Variables must reach the container. Verify GDR once with `NCCL_DEBUG=INFO`
  (`GDRDMA` and `GDR 1`), then use `NCCL_DEBUG=WARN` for performance runs.
- Use ABI-4 libionic and `/dev/infiniband`. Prefer `iommu=pt` nodes on Crusoe,
  although the warning alone does not mean GDR is disabled.
- Never share a writable Inductor cache across ranks. If one rank fails first,
  inspect its Docker service before treating later TCPStore errors as the cause.

Detailed communication results:
`/shared_nfs/zirui/runs/allgather-debug-20260916/RESULTS.md`.
