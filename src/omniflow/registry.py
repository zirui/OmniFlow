"""Simple registries for model/dataset/trainer builders."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict, Tuple

MODEL_REGISTRY: Dict[str, Callable[[dict], Any]] = {}
DATASET_REGISTRY: Dict[str, Callable[[dict], Tuple[Any, Any]]] = {}
TRAINER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    def decorator(fn: Callable[[dict], Any]):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def register_dataset(name: str):
    def decorator(fn: Callable[[dict], Tuple[Any, Any]]):
        DATASET_REGISTRY[name] = fn
        return fn

    return decorator


def register_trainer(name: str):
    def decorator(fn: Callable[..., Any]):
        TRAINER_REGISTRY[name] = fn
        return fn

    return decorator


def get_model_builder(name: str) -> Callable[[dict], Any]:
    if name not in MODEL_REGISTRY:
        for module_name in (
            f"omniflow.models.registrations.{name}",
            f"models.registrations.{name}",
        ):
            try:
                import_module(module_name)
            except ModuleNotFoundError:
                continue
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name]


def get_dataset_builder(name: str) -> Callable[[dict], Tuple[Any, Any]]:
    if name not in DATASET_REGISTRY:
        for module_name in (
            f"omniflow.data.registrations.{name}",
            f"data.registrations.{name}",
        ):
            try:
                import_module(module_name)
            except ModuleNotFoundError:
                continue
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset name: {name}")
    return DATASET_REGISTRY[name]


def get_trainer_builder(name: str) -> Callable[..., Any]:
    if name not in TRAINER_REGISTRY:
        for module_name in (f"omniflow.trainers.{name}", f"trainers.{name}"):
            try:
                import_module(module_name)
            except ModuleNotFoundError:
                continue
    if name not in TRAINER_REGISTRY:
        raise KeyError(f"Unknown trainer name: {name}")
    return TRAINER_REGISTRY[name]