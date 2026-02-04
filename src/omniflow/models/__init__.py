
from .wan import (
    WanVideoConfig,
    WanVideoForConditionalGeneration,
    WanVideoProcessor
)

# TODO: zirui, temporary for debugging
from .wan_new import WanVideoConfig as WanVideoConfig_new
from .wan_new import WanVideoForConditionalGeneration as WanVideoForConditionalGeneration_new
from .wan_new import WanVideoProcessor as WanVideoProcessor_new


__all__ = [
    "WanVideoConfig",
    "WanVideoForConditionalGeneration",
    "WanVideoProcessor",
    "WanVideoConfig_new",
    "WanVideoForConditionalGeneration_new",
    "WanVideoProcessor_new"
]