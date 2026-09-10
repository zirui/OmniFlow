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
environment. The same pattern also works over direct SSH: start all commands
concurrently, replace `spur run ... env` with `ssh <node> env`, and keep the
same rendezvous values plus one unique contiguous `NODE_RANK` per node. Use
only nodes owned by the current allocation.

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

## Network checks

The four-node profile includes the Primus runner scheduling defaults that
matter to native `torchrun`: `HSA_ENABLE_SDMA=1`, `GPU_MAX_HW_QUEUES=2`,
`TORCH_NCCL_HIGH_PRIORITY=1`, `NCCL_CHECKS_DISABLE=1`,
`NCCL_P2P_NET_CHUNKSIZE=524288`, and the tensor-register allocator hook off.
Keep these defaults for both launch styles unless a matched A/B test shows a
regression. `run_with_docker.sh` also applies the shared AINIC defaults used by
Primus CLI: all eight `ionic_*:1` HCAs, GID 1, TC/FIFO TC 104/192, RoCE v2,
inline sends, one QP per connection, retry/timeout 20/300, 56 P2P channels,
`librccl-anp.so`, GDR flush disabled, CPU affinity ignored, and cross-NIC off.
Only the socket interface and DMA-BUF setting differ below.

For multi-node runs, verify that the ABI-4 libionic mount and
`/dev/infiniband` exist. Prefer nodes with `iommu=pt` in `/proc/cmdline`;
without it RCCL emits a stability warning, but the qualified DMA-BUF path has
also completed GDRDMA training on nodes that lack it. With `NCCL_DEBUG=INFO`,
confirm that channels use
`NET/RCCL-ANP/.../GDRDMA` and rings report `GDR 1`. The qualified cluster
overrides are:

| Cluster | Socket interface | DMA-BUF | Notes |
|---|---|---:|---|
| Crusoe | `ens3` | `1` (default) | RCCL-ANP; `NCCL_NET_GDR_LEVEL=SYS`, `NCCL_NET_GDR_READ=1` |
| DCCS | `fenic` | `0` | Set `NCCL_DMABUF_ENABLE=0` and both socket-interface variables |

Set `NCCL_IB_DISABLE=1` only for a host-staged comparison. Pass the cluster
overrides to every per-node command when launching through separate
allocations or direct SSH.

Common failures:

- `expected GBS`: allocation size does not match the selected profile.
- Rendezvous timeout: a rank is missing/duplicated or cannot reach rank 0.
- `Missing "iommu=pt"`: prefer another node when available; this warning alone
  does not mean the DMA-BUF GDR path failed.
- Docker name conflict: remove the stale container or set `CONTAINER_NAME`.
- Empty/invalid cache metadata: rebuild the cache locally; do not share a
  writable cache directory across ranks.
