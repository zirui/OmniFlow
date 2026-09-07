# FLUX.1-Schnell MLPerf training

This recipe runs FLUX.1-Schnell directly with PyTorch FSDP2 and no Primus CLI.
The MXFP4 profiles use the Primus-Turbo package supplied by their pinned image.

The qualified profiles target MI355X GPUs and the MLPerf validation-loss
threshold `0.586`. Tensorwise FP8 uses `zirui3/primus-v26.3-flux:v0.4`; the
DP32 PR492 run uses `zirui3/primus-v26.3-flux:v0.4.1`.

The two accepted MXFP4 Pareto recipes use
`zirui3/primus-v26.3-flux:v0.4-mxfp4-uos`. Its reproducible source is
[`Dockerfile.mxfp4`](Dockerfile.mxfp4), which pins Primus-Turbo revision
`220ead50861860f47161787694319f4b0853df3a` and the `uos_7p25` scale policy.
Training runs directly through OmniFlow without the Primus CLI or Python
framework. Rebuild the validated image with:

```bash
mkdir -p /tmp/empty-context
docker build -f examples/mlperf/flux1/Dockerfile.mxfp4 \
  -t zirui3/primus-v26.3-flux:v0.4-mxfp4-uos /tmp/empty-context
```

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
| `config_4n_gbs1024_mxfp4_pareto_a.sh` | 4 | 32 | 1 | 1024 |
| `config_4n_gbs1024_mxfp4_pareto_b.sh` | 4 | 32 | 1 | 1024 |

Run inside a matching Slurm allocation:

```bash
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
FLUX_CONFIG=config_2n_gbs1024.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

The MXFP4 profiles expose only the accepted recipes:

- `pareto_a`: 76 BF16 and 152 MXFP8 forward linears.
- `pareto_b`: 19 BF16, 38 MXFP4, and 171 MXFP8 forward linears.

Both route all 228 selected block-linear backward paths through MXFP4. They
require the pinned MXFP4 image and set the required AITER preshuffle backend.

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
