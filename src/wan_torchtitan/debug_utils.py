import logging

import torch

logger = logging.getLogger(__name__)
def print_tensor(t : torch.Tensor, name : str):
    print("<------------{}".format(name))
    if torch.is_tensor(t):
        print("shape={}, dtype={}, device={}, contiguous={}".format(t.shape, t.dtype, t.device, t.is_contiguous()))
        flat = t.flatten()
        print("value : {} ... {}".format(flat[:10], flat[-10:]))
    else:
        print("NOT A TENSOR")
    print("{}------------>\n".format(name), flush=True)