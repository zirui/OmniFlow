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
MAX_STEPS=20 MLPERF_ENABLE=true SAVE_STRATEGY=none \
DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
FLUX_CONFIG=config_1n_dp8_fp8_allgather.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh

# Place cache-node0.tar.zst ... cache-node3.tar.zst under OUTPUT_ROOT.
TORCHINDUCTOR_CACHE_SEED=/output/cache-node%r.tar.zst \
DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
FLUX_CONFIG=config_4n_dp8_fp8_allgather.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

On homogeneous MI355X nodes, a verified archive may be copied to another node
rank. Never let multiple ranks write one shared cache. Rebuild the cache when
the image, PyTorch/Triton/FlyDSL versions, model graph, batch shapes, or compile
settings change.

## Network checks

For multi-node runs, verify that the ABI-4 libionic mount and
`/dev/infiniband` exist. With `NCCL_DEBUG=INFO`, confirm that channels use
`NET/RCCL-ANP/.../GDRDMA`. Set `NCCL_IB_DISABLE=1` only for a host-staged
comparison.

Common failures:

- `expected GBS`: allocation size does not match the selected profile.
- Rendezvous timeout: a rank is missing/duplicated or cannot reach rank 0.
- Docker name conflict: remove the stale container or set `CONTAINER_NAME`.
- Empty/invalid cache metadata: rebuild the cache locally; do not share a
  writable cache directory across ranks.
