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

## TODO:
1. **Parallism**
  - [ ] CP
  - [ ] Ulysses
  - [ ] USP
2. **Models**
   - [ ] hunyuan-video
3. **Sparse Attention**
    - [ ] BSA
    - [ ] SA
4. **Backend**
   - [ ] native pytorch trainer
   - [ ] torchtitan
5. **Data**
  - [ ] video streaming input
6. **Evaluaton**
  * [ ] integreted video-gen eval


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

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
