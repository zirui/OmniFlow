# FLUX.1-Schnell MLPerf training

This recipe runs FLUX.1-Schnell directly with PyTorch FSDP2 and no Primus CLI.
The MXFP4 profiles use the Primus-Turbo package supplied by their pinned image.

The qualified profiles target MI355X GPUs and the MLPerf validation-loss
threshold `0.586`. Tensorwise FP8 uses `zirui3/primus-v26.3-flux:v0.4`; the
DP32 PR492 run uses `zirui3/primus-v26.3-flux:v0.4.1`.

The two MXFP4 Pareto recipes use
`zirui3/primus-v26.3-flux:v0.4-mxfp4-mixed-quant-uos`. Its reproducible source is
[`Dockerfile.mxfp4`](Dockerfile.mxfp4), which pins Primus-Turbo revision
`220ead50861860f47161787694319f4b0853df3a`, applies the committed mixed-quant
patch `a58e3d9e1ea9602ebf2f3d4553d8e6bffaa21030`, and retains the `uos_7p25`
scale policy.
Training runs directly through OmniFlow without the Primus CLI or Python
framework. Build the updated image from the OmniFlow repository root so the
Docker context includes the pinned patch:

```bash
docker build -f examples/mlperf/flux1/Dockerfile.mxfp4 \
  -t zirui3/primus-v26.3-flux:v0.4-mxfp4-mixed-quant-uos .
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
| `config_4n_gbs1024_mxfp4_custom.sh` | 4 | 32 | 1 | 1024 |
| `config_4n_gbs1024_mxfp4_under80.sh` | 4 | 32 | 1 | 1024 |

Run inside a matching Slurm allocation:

```bash
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
FLUX_CONFIG=config_2n_gbs1024.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

The qualified `pareto_a` and `pareto_b` recipes remain fixed: `pareto_a` has
76 BF16 and 152 MXFP8 forward linears; `pareto_b` has 19 BF16, 38 MXFP4, and
171 MXFP8 forward linears. The `custom` recipe is an experimental composition
surface. All three keep MXFP4 backward on all 228 selected block linears and
require the pinned image and AITER preshuffle backend.

### Qualified sub-80-minute profile

`config_4n_gbs1024_mxfp4_under80.sh` is the performance-qualified extension of
Pareto A: all 152 double-block Linear modules use BF16 forward, the 76
single-block Linear modules use MXFP8 forward, and all 228 backward paths remain
MXFP4. It uses tensor-scale E4M3 HSDP gradient AllReduce, seed 10009, delayed
validation, mixed operand quantization, and a recipe-specific exact Inductor
cache. On four MI355X nodes it reached `0.585692 @ step 8704` in **79.882 min**
(E2E 1859.60 samples/s, steady median 1919.93 samples/s):

```bash
TORCHINDUCTOR_CACHE_SEED=/path/to/under80-cache-rank%r.tar.zst \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
FLUX_CONFIG=config_4n_gbs1024_mxfp4_under80.sh \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

The exact cache is mandatory for the qualified timing and must be rebuilt after
any image, graph, shape, or compiler change.

### Custom MXFP4 options

| Environment variable | Values |
|---|---|
| `FLUX_MXFP4_RECIPE` | `custom` (or fixed `pareto_a` / `pareto_b`) |
| `FLUX_MXFP4_FORWARD_PRECISION` | `bf16`, `mxfp8`, `mxfp4` |
| `FLUX_MXFP4_BF16_FORWARD_SCOPE` | `none`, `double_img_up_2_3`, `double_img_up`, `double_img_mlp_attn_out`, `double_img_all`, `double_img_all_txt_mlp_down`, `double_img_all_txt_mlp`, `double_all` |
| `FLUX_MXFP4_SELECTIVE_FORWARD_SCOPE` | `none`, `single_linear2`, `single_linear1_early`, `late_txt_mlp_up` |
| `FLUX_MXFP4_FORWARD_HADAMARD` | `none`, `mlp`, `all` |
| `FLUX_MXFP4_ACTIVATION_RESIDUAL` | `none`, `double_img_up_0_1`, `double_img_up_2_3`, `double_img_up_0_3`, `double_img_up_4_8`, `double_img_up_early`, `double_img_up_late`, `double_img_up`, `double_txt_up`, `double_up`, `up`, `mlp`, `all` |
| `FLUX_MXFP4_ACTIVATION_RESIDUAL_DTYPE` | `bf16`, `mxfp8`, `mxfp4` |
| `FLUX_MXFP4_EVAL_PRECISION` | `same`, `bf16`; Pareto profiles use `bf16` |
| `FLUX_MXFP4_GRADIENT_SR` | `false`, `true` |
| `FLUX_MXFP4_CAPTURE_STEPS` | comma-separated optimizer steps |
| `FLUX_MXFP4_CAPTURE_MODULES` | comma-separated exact module FQNs |
| `FLUX_MXFP4_CAPTURE_DIR` | capture output directory; defaults to `$OUTPUT_DIR/mxfp4-captures` |
| `FLUX_MXFP4_BF16_FORWARD_SWITCH_STEP` | positive threshold step, or `0` to disable |

Scope membership for the 19-double/38-single-block model is:

| Override | Scope | Count | Members |
|---|---|---:|---|
| BF16 | `double_img_up_2_3` | 2 | image MLP-up in double blocks 2–3 |
| BF16 | `double_img_up` | 19 | all image MLP-up |
| BF16 | `double_img_mlp_attn_out` | 57 | image MLP up/down and attention output |
| BF16 | `double_img_all` | 76 | previous image modules plus image QKV |
| BF16 | `double_img_all_txt_mlp_down` | 95 | previous 76 plus text MLP-down |
| BF16 | `double_img_all_txt_mlp` | 114 | previous 95 plus text MLP-up |
| BF16 | `double_all` | 152 | every double-block Linear |
| MXFP4 | `single_linear2` | 38 | every single-block `linear2` |
| MXFP4 | `single_linear1_early` | 57 | previous 38 plus `linear1` in single blocks 0–18 |
| MXFP4 | `late_txt_mlp_up` | 48 | base 38 `linear2` plus text MLP-up in double blocks 9–18 |

The selective MXFP4 names are cumulative as shown: both larger scopes include
all 38 `single_linear2` members, then add their named tier.

Precision resolution is deterministic: the BF16 scope wins, then the selective
MXFP4 scope, then the base precision. Hadamard and residual correction only run
on modules resolved to MXFP4. On overlap residual wins over Hadamard; disjoint
selections combine. Hadamard applies the same rowwise RHT to activation and
weight. Residual correction computes `input - dequant(Q4(input))`; its MXFP4
mode reuses the forward FP4 weight. `bf16` evaluation bypasses low-precision
forward without changing the training strategy. Healing changes eligible
MXFP4-forward wrappers to BF16 once on the first forward at or after the
configured threshold (including resumed runs) and leaves backward unchanged.

Captures are deduplicated by step and FQN and saved only on rank zero. Analyze
them with `tools/mxfp4_real_tensor_gate.py` or
`tools/mxfp4_residual_decomposition.py` (both provide `--help`).

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
