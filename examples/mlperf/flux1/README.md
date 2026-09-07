# FLUX.1-Schnell MLPerf training

This recipe runs FLUX.1-Schnell directly with PyTorch FSDP2. It has no Primus
CLI or Primus Python-package dependency.

The qualified profiles target MI355X GPUs, tensorwise FP8, and the MLPerf
validation-loss threshold `0.586`. They use the validated image:

```bash
docker pull zirui3/primus-v26.3-flux:v0.4
```

The image provides the optional `primus_turbo` FlyDSL kernels; training still
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

For the fastest four-node path, set `FLUX_FP8_ALL_GATHER=1` and provide
node-local exact max-autotune caches with
`TORCHINDUCTOR_CACHE_SEED=/output/cache-node%r.tar.zst`. `%r` resolves to
`NODE_RANK`; no additional profile file is required.

`run_with_docker_slurm.sh` starts one container per node;
`run_with_docker.sh` starts one `torchrun` worker per GPU. Set `DOCKER_IMAGE`
only to test another compatible image.

See [README-dev.md](README-dev.md) for rendezvous, cache prewarming, and
multi-allocation operation.
