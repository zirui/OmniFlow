"""
wan_new2: Components + TaskPipeline design (native PyTorch/FSDP).

Goal:
- Keep modeling close to official Wan2.2 (pure torch modules).
- Keep trainer generic: trainer calls model(batch, scheduler) -> {"loss": ...}.
- Decouple training/inference workflow (pipeline) from modeling (modules).
"""

from .adapter import WanNew2ForTraining

__all__ = [
    "WanNew2ForTraining",
]