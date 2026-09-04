###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Explicit factory registry for diffusion model, dataset, and trainer builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MODEL_BUILDERS: dict[str, Callable[[dict], Any]] = {}
DATASET_BUILDERS: dict[str, Callable[[dict], tuple[Any, Any]]] = {}
TRAINER_BUILDERS: dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    def decorator(fn):
        MODEL_BUILDERS[name] = fn
        return fn

    return decorator


def register_dataset(name: str):
    def decorator(fn):
        DATASET_BUILDERS[name] = fn
        return fn

    return decorator


def register_trainer(name: str):
    def decorator(fn):
        TRAINER_BUILDERS[name] = fn
        return fn

    return decorator


def _build_wan_model(model_config: dict):
    from omniflow.models.registrations.wan import build_wan_model

    return build_wan_model(model_config)


def _build_wan_dataset(dataset_config: dict):
    from omniflow.data.registrations.wan import build_wan_dataset

    return build_wan_dataset(dataset_config)


def _build_flux_model(model_config: dict):
    from omniflow.models.registrations.flux import build_flux_model

    return build_flux_model(model_config)


def _build_flux_preset_model(preset: str):
    def _builder(model_config: dict):
        from omniflow.models.registrations.flux import build_flux_model

        config = dict(model_config)
        config["model_preset"] = preset
        return build_flux_model(config)

    return _builder


def _build_flux_dataset(dataset_config: dict):
    from omniflow.data.registrations.flux import build_flux_dataset

    return build_flux_dataset(dataset_config)


def _build_fsdp2_trainer(
    *, model, dataset, processor, trainer_args: dict, eval_dataset=None, eval_processor=None
):
    from omniflow.trainers.fsdp2 import build_fsdp2_trainer

    return build_fsdp2_trainer(
        model=model,
        dataset=dataset,
        processor=processor,
        eval_dataset=eval_dataset,
        eval_processor=eval_processor,
        trainer_args=trainer_args,
    )


MODEL_BUILDERS.update(
    {
        "flux": _build_flux_model,
        "flux.1-dev": _build_flux_preset_model("flux.1-dev"),
        "flux.1-schnell": _build_flux_preset_model("flux.1-schnell"),
        "wan": _build_wan_model,
    }
)
DATASET_BUILDERS.update({"flux": _build_flux_dataset, "wan": _build_wan_dataset})
TRAINER_BUILDERS.update({"fsdp2": _build_fsdp2_trainer})


def get_model_builder(name: str) -> Callable[[dict], Any]:
    try:
        return MODEL_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown model name: {name}") from None


def get_dataset_builder(name: str) -> Callable[[dict], tuple[Any, Any]]:
    try:
        return DATASET_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset name: {name}") from None


def get_trainer_builder(name: str) -> Callable[..., Any]:
    try:
        return TRAINER_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown trainer name: {name}") from None
