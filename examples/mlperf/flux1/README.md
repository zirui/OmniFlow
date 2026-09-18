# FLUX.1-Schnell MLPerf training

This recipe runs FLUX.1-Schnell directly with PyTorch FSDP2. It has no Primus
CLI or Primus Python-package dependency.

The qualified profiles target MI355X GPUs, tensorwise FP8, and the MLPerf
validation-loss threshold `0.586`. The base image is
`zirui3/primus-v26.3-flux:v0.4.3`; the DP32 PR492 run uses
`zirui3/primus-v26.3-flux:v0.4.1`.

Both images provide the optional `primus_turbo` FlyDSL kernels; training still
runs directly through OmniFlow without the Primus CLI or Python framework.

## Data

Download the preprocessed MLPerf datasets:

```bash
mkdir -p /path/to/data
cd /path/to/data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri
```

The data root must contain:

```text
cc12m_preprocessed/
coco_preprocessed/
empty_encodings/
```

## Run

Each profile fixes the node count, micro-batch size, accumulation, and global
batch size:

| Profile | Nodes | MBS | GA | GBS |
|---|---:|---:|---:|---:|
| `config_1n_gbs512.sh` | 1 | 64 | 1 | 512 |
| `config_1n_gbs1024.sh` | 1 | 32 | 4 | 1024 |
| `config_2n_gbs1024.sh` | 2 | 32 | 2 | 1024 |
| `config_4n_gbs1024.sh` | 4 | 32 | 1 | 1024 |

Run inside a matching Slurm allocation:

```bash
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
FLUX_CONFIG=config_2n_gbs1024.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

For a one-step smoke test:

```bash
MAX_STEPS=1 SAVE_STRATEGY=none MLPERF_CLEAR_CACHES=false \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
FLUX_CONFIG=config_1n_gbs512.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

Enable native FP8 parameter AllGather with `FLUX_FP8_ALL_GATHER=1`; no
additional profile file is required. See [README-dev.md](README-dev.md) for the
qualified DP32 + PR492 command and node-local exact-cache setup.

`run_with_docker_slurm.sh` starts one container per node;
`run_with_docker.sh` starts one `torchrun` worker per GPU. Set `DOCKER_IMAGE`
only to test another compatible image.

See [README-dev.md](README-dev.md) for rendezvous, cache prewarming, and
multi-allocation operation.

## Cached max-autotune

Generate one node-local cache per allocation node before training. The setup
supports 1-node, 2-node, and 4-node allocations:

```bash
ALLOCATION_JOB_ID=<job-id> DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/shared/path/to/output \
bash examples/mlperf/flux1/setup_max_autotune_cache.sh
```

Then reuse the same `OUTPUT_ROOT` for training:

```bash
FLUX_CONFIG=config_4n_gbs1024.sh \
TORCHINDUCTOR_CACHE_SEED=/output/cache-node%r.tar.zst \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/shared/path/to/output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

The launcher mounts host `OUTPUT_ROOT` at `/output` in the container, so
`/output/cache-node%r.tar.zst` maps to `$OUTPUT_ROOT/cache-node%r.tar.zst`;
`%r` is the node rank. Rebuild caches after changing the image, compiler, model
graph, batch shapes, or compile options.

## One-node hipBLASLt tuning

The GBS1024 one-node profile defaults to `hipblaslt_fixed`. It retains the eight
qualified FlyDSL shapes and pins hipBLASLt 1.4.1 solutions for the two remaining
large E5M2-by-E4M3 GEMMs, `(16384, 3072, 21504)` and
`(21504, 3072, 16384)`. The extension rejects other hipBLASLt versions and GPU
architectures because solution indices are build- and architecture-specific.
Set `FLUX_FP8_GEMM_BACKEND=selective_flydsl` to run the untuned control.

In isolated validation, the selected kernels were 7.2% and 9.1% faster than
`at::_scaled_mm`. Matched 60-step GBS1024 runs on two allocated MI355X nodes
measured the optimized profile at 62.35 and 63.46 samples/GPU/s. On the latter node, the
original profile measured 54.76 samples/GPU/s over the same steady-state step
window, a 15.9% end-to-end improvement. The one-node profile includes the
qualified FSDP/RCCL scheduling settings used in that measurement.
