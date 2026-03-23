# Training
A flexible training framework for multi-modality. 

## Features

- **Flexible**: Supports training from scratch or fine-tuning from pretrained checkpoints.
- **Configurable**: Simple YAML configuration.




- Models

| model  | deepspeed  | FSDP  |CP  | Ulysses  |USP|
|---|---|---|---|---|--|
| Wan2  | ✅  | ✅  | ❌  | ❌ |❌ |
| Hunyuan-video  |   |   |   |   ||


## difussion training
```
Text ─▶ Encoder ─┐
                  ├─▶ Denoiser(z_t, t) ─▶ Loss ─▶ Optimizer
Video ─▶ VAE ─▶ z_t
```


## TODO:
1. **Parallism**
  - [ ] CP
  - [x] Ulysses
  - [ ] USP
2. **Models**
   - [ ] hunyuan-video
3. **Attention**
    - [x] FA2/FA3(aiter-FA-v3)/FlexAttention
    - [ ] NABLA
    - [ ] STA
    - [ ] VSA
    - [ ] SSA
    - [ ] BSA
    - [ ] SLA
4. **Backend**
   - [x] native pytorch trainer
   - [x] torchtitan
5. **Data**
  - [ ] video streaming input
6. **Evaluaton**
  * [ ] integreted video-gen eval


## Installation

1. Install dependencies:
```bash
# Update submodules
git submodule update --init --recursive

# Install general dependencies with default (no specific accelerator)

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Install with CUDA/ROCm 7.0 support
uv sync --extra rocm

# Install with ROCm 7.1 support
uv sync --extra rocm7-1

# Install with CUDA support (cu128)
uv sync --extra cuda

# Install with torchtitan (can be combined with other extras)
uv sync --extra torchtitan --extra rocm
```

2. Download pretrained models(if needed):
```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B Wan2.1_VAE.pth --local-dir ./checkpoints --local-dir-use-symlinks False
huggingface-cli download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth --local-dir ./checkpoints --local-dir-use-symlinks False
```

## Directory Structure


```text
.
|-- notebooks
|-- src
|   |-- omniflow
|   |   |-- configs
|   |   |-- data
|   |   |-- models
|   |   |-- training
|   |   `-- utils
|   `-- wan_torchtitan
|       |-- model
|       `-- train_configs
`-- third_party
    `-- torchtitan
```


- `models/`: model definitions.
- `data/`: Data processing and dataset loading.
- `training/`: Custom trainer and flow-matching scheduler.
- `configs/`: Training configurations.
- `train.py`: Main training script.
- `run.sh`: Launch script for distributed training.

## Usage

### Data Preparation

Prepare your video dataset in JSONL format:

```json
{"video": "/path/to/video1.mp4", "prompt": "A beautiful sunset"}
{"video": "/path/to/video2.mp4", "prompt": "A running dog"}
```

### Training

**Single GPU:**

```bash
python train.py --config configs/wan2.2_t2v_5b.yaml


# Run training based on torchtitan(currently need to manually add wan to torchtitan/experiments/__init__.py)

# copy wan_torchtitan to torchtitan/experiments/wan
# cp -r src/wan_torchtitan torchtitan/experiments/wan
ln -s ../../../../src/wan_torchtitan third_party/torchtitan/torchtitan/experiments/wan


# Add wan to torchtitan/experiments/__init__.py
torchtitan/experiments/__init__.py
 _supported_experiments = frozenset(
-    ["flux", "simple_fsdp.llama3", "simple_fsdp.deepseek_v3", "vlm"]
+    ["flux", "simple_fsdp.llama3", "simple_fsdp.deepseek_v3", "vlm", "wan"]
 )

cd third_party/torchtitan && bash -x torchtitan/experiments/wan/run_train.sh
```

**Multi-GPU (Distributed):**


```bash
# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12356 \
  train.py --config ${CONFIG}
```
