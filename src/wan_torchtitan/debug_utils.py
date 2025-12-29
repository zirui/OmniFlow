import logging

import torch

logger = logging.getLogger(__name__)
def print_tensor(t : torch.Tensor, name : str):
    logger.info("<------------{}".format(name))
    if torch.is_tensor(t):
        logger.info("shape={}, dtype={}, device={}, contiguous={}".format(t.shape, t.dtype, t.device, t.is_contiguous()))
    else:
        logger.info("NOT A TENSOR")
    logger.info("{}------------>".format(name))