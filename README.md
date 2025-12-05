# Training
A flexible training framework for multi-modality. 

## Features

- **Flexible**: Supports training from scratch or fine-tuning from pretrained checkpoints.
- **Configurable**: Simple YAML configuration.

- Models

| model  | deepspeed  | FSDP  |HSDP| cp  | ulysses  |
|---|---|---|---|---|--|
| wan2  | y  |   |   |   ||
| hunyuan-video  |   |   |   |   ||

## TODO:
- parallism 
    - [ ] FSDP
    - [ ] CP
    - [ ] BSA
- Models
    - [ ] hunyuan-video
- Block Sparse Attention
- Backend
  - [ ]torchtitan


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
