# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# TODO: zirui, add validation and inference config

from dataclasses import dataclass, field


@dataclass
class Training:
    classifier_free_guidance_prob: float = 0.0
    """Classifier-free guidance with probability `p` to dropout each text encoding independently.
    If `n` text encoders are used, the unconditional model is trained in `p ^ n` of all steps.
    For example, if `n = 2` and `p = 0.447`, the unconditional model is trained in 20% of all steps"""
    img_size: int = 256
    """Image width to sample"""
    test_mode: bool = False
    """Whether to use integration test mode, which will randomly initialize the encoder and use a dummy tokenizer"""


@dataclass
class Encoder:
    t5_checkpoint_path: str = "google/umt5-xxl"
    """T5 encoder to use, HuggingFace model name. This field could be either a local folder path,
        or a Huggingface repo name."""
    vae_checkpoint_path: str = (
        "torchtitan/experiments/wan/assets/autoencoder/Wan2.1_VAE.pth"
    )
    vae_type: str = "wan_video_vae_38"
    # """Autoencoder checkpoint path to load. This should be a local path referring to a safetensors file."""
    # max_t5_encoding_len: int = 256
    # """Maximum length of the T5 encoding."""


@dataclass
class Validation:
    enable_classifier_free_guidance: bool = False
    """Whether to use classifier-free guidance during sampling"""
    classifier_free_guidance_scale: float = 5.0
    """Classifier-free guidance scale when sampling"""
    denoising_steps: int = 50
    """How many denoising steps to sample when generating an image"""
    eval_freq: int = 100
    """Frequency of evaluation/sampling during training"""
    save_img_count: int = 1
    """ How many images to generate and save during validation, starting from
    the beginning of validation set, -1 means generate on all samples"""
    save_img_folder: str = "img"
    """Directory to save image generated/sampled from the model"""
    all_timesteps: bool = False
    """Whether to generate all stratified timesteps per sample or use round robin"""


@dataclass
class Inference:
    """Inference configuration"""

    save_img_folder: str = "inference_results"
    """Path to save the inference results"""
    prompts_path: str = "./torchtitan/experiments/flux/inference/prompts.txt"
    """Path to file with newline separated prompts to generate images for"""
    local_batch_size: int = 2
    """Batch size for inference"""
    img_size: int = 256
    """Image size for inference"""


@dataclass
class JobConfig:
    """
    Extend the tyro parser with custom config classes for Flux model.
    """

    training: Training = field(default_factory=Training)
    encoder: Encoder = field(default_factory=Encoder)
    # validation: Validation = field(default_factory=Validation)
    # inference: Inference = field(default_factory=Inference)
