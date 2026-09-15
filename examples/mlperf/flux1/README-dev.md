# FLUX multi-node development notes

Start from the OmniFlow repository root:

```bash
REPO=/shared_nfs/zirui/code/OmniFlow
DATA_ROOT=/shared_nfs/zirui/data
OUTPUT_ROOT=/shared_nfs/zirui/runs/flux-$(date -u +%Y%m%dT%H%M%SZ)
DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4
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

The qualified four-node DP32 native-FP8-AllGather run uses the v0.4.1 image,
which adds Primus-Turbo PR492 to v0.4:

```bash
DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4.1 \
FLUX_CONFIG=config_4n_gbs1024.sh \
DP_REPLICATE=1 FLUX_FP8_ALL_GATHER=1 \
TORCH_COMPILE_MODE=default LOG_FREQ=1 \
NCCL_DMABUF_ENABLE=0 \
NCCL_SOCKET_IFNAME=fenic GLOO_SOCKET_IFNAME=fenic \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

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
environment.

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

MXFP4 caches must be built with the MXFP4 image and exact recipe. Pareto A,
Pareto B, custom recipes, and tensorwise FP8 have different graphs and cannot
share archives. For example, prewarm Pareto A with:

```bash
DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4-mxfp4-mixed-quant-uos
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=20G \
  -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/workspace/OmniFlow/src \
  -e TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR" \
  -e FLUX_FLOAT8_RECIPE= -e FLUX_MXFP4_RECIPE=pareto_a \
  -e FLUX_MXFP4_EVAL_PRECISION=bf16 \
  -e PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER -e PRIMUS_TURBO_AUTO_TUNE=0 \
  -v "$REPO:/workspace/OmniFlow" -v /shared_nfs:/shared_nfs \
  -w /workspace/OmniFlow "$DOCKER_IMAGE" \
  python examples/mlperf/flux1/prewarm_inductor_cache.py
```

On homogeneous MI355X nodes, a verified archive may be copied to another node
rank. Never let multiple ranks write one shared cache. Rebuild the cache when
the image, recipe, PyTorch/Triton/FlyDSL versions, model graph, batch shapes,
or compile settings change.

## Optimization compatibility

| Optimization | MXFP4 Pareto A/B |
|---|---|
| DP32 topology and MI355X scheduling | Inherited from `config_4n_gbs1024.sh` |
| Qualified Crusoe NCCL settings | Inherited; override `FLUX_NCCL_*` off Crusoe |
| Cached Inductor max-autotune | Supported with a separate exact cache per recipe |
| Native FP8 parameter AllGather / PR492 | Not supported by `MXFP4Linear`; requires a new FSDP parameter transport |
| TorchAO FP8 input/weight reuse and selective FlyDSL | FP8-only; not used by MXFP4 recipes |

The fixed MXFP4 profiles force `FLUX_FP8_ALL_GATHER=0` and keep
`DP_REPLICATE=4`. Do not substitute the PR492 `v0.4.1` image: the MXFP4 image
has a separate pinned Primus-Turbo revision and UOS scale policy. A combined
image or compressed MXFP4 AllGather needs independent correctness and
performance qualification.

## Network checks

For multi-node runs, verify that the ABI-4 libionic mount and
`/dev/infiniband` exist. With `NCCL_DEBUG=INFO`, confirm that channels use
`NET/RCCL-ANP/.../GDRDMA`. Set `NCCL_IB_DISABLE=1` only for a host-staged
comparison. DCCS runs should set `FLUX_NCCL_DMABUF_ENABLE=0`,
`NCCL_SOCKET_IFNAME=fenic`, and `GLOO_SOCKET_IFNAME=fenic`.

Common failures:

- `expected GBS`: allocation size does not match the selected profile.
- Rendezvous timeout: a rank is missing/duplicated or cannot reach rank 0.
- Docker name conflict: remove the stale container or set `CONTAINER_NAME`.
- Empty/invalid cache metadata: rebuild the cache locally; do not share a
  writable cache directory across ranks.
