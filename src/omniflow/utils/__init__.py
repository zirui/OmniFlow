"""Shared OmniFlow utilities."""

__all__ = [
    "extract_vision_info",
    "fetch_image",
    "fetch_video",
    "process_vision_info",
    "smart_resize",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import vision_process

    return getattr(vision_process, name)
