# TODO: zirui, add validation and inference config for wan_new2

from dataclasses import dataclass, field


@dataclass
class Training:
    # Keep compatibility with wan_torchtitan: optionally load DiT weights from a pretrained folder/file
    load_from_pretrained_path: str = ""


@dataclass
class Encoder:
    # Encoder checkpoints (UMT5 encoder-only + Wan VAE)
    t5_checkpoint_path: str = "google/umt5-xxl"
    vae_checkpoint_path: str = ""
    # "wan2.1" or "wan2.2" (affects VAE z_dim and wrapper selection)
    vae_type: str = "wan2.1"


@dataclass
class Wan:
    # Task type for Wan DiT: t2v / i2v / ti2v / s2v
    model_type: str = "t2v"


@dataclass
class JobConfig:
    """
    Extend TorchTitan JobConfig with Wan-specific knobs.

    This class is merged into `torchtitan.config.JobConfig` via
    `--job.custom_config_module=<this module>`.
    """

    training: Training = field(default_factory=Training)
    encoder: Encoder = field(default_factory=Encoder)
    wan: Wan = field(default_factory=Wan)