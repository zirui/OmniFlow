from model.wan_video_text_encoder import WanTextEncoder

import random
import torch

from transformers import UMT5EncoderModel, AutoConfig

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

DEFAULT_DEVICE = torch.device("cuda:0")
CKPT_PATH_LOCAL = "/root/zirui/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"

def main(args):
    torch.cuda.set_device(DEFAULT_DEVICE)

    input = torch.randint(
        low=100,
        high=30000,
        size=(4, 1024),
        dtype=torch.int64,
        device=DEFAULT_DEVICE,
        )

    model_local = WanTextEncoder().to(DEFAULT_DEVICE)
    state_local = torch.load(CKPT_PATH_LOCAL, map_location=DEFAULT_DEVICE)
    model_local.load_state_dict(state_local, strict=True)
    model_local.eval()
    with torch.no_grad():
       print("forward model_local ...", flush=True)
       output_local = model_local(input)
    print("output_local, shape={}, dtype={}, values={}".format(output_local.shape, output_local.dtype, output_local), flush=True)

    # config = AutoConfig.from_pretrained("google/umt5-xxl")
    # model_hf = UMT5EncoderModel.from_config(config)
    model_hf = UMT5EncoderModel.from_pretrained("google/umt5-xxl", device_map=DEFAULT_DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        print("forward model_hf ...", flush=True)
        output_hf = model_hf(input).last_hidden_state
    print("output_hf, shape={}, dtype={}, values={}".format(output_hf.shape, output_hf.dtype, output_hf), flush=True)

if __name__ == "__main__":
    args = None
    main(args)
