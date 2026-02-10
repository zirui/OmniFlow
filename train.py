#!/usr/bin/env python3
"""Unified training entry point (registry-based CLI)."""

import argparse
import logging

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _require_dict(obj: object, name: str) -> dict:
    if not isinstance(obj, dict):
        raise SystemExit(f"Config '{name}' must be a dict, got: {type(obj).__name__}")
    return obj


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("omniflow")

    parser = argparse.ArgumentParser(description="OmniFlow Unified Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args(argv)

    from omniflow.attention import set_attention_backend
    from omniflow.registry import get_dataset_builder, get_model_builder, get_trainer_builder

    logger.info(f"Loading config from: {args.config}")
    config = _require_dict(load_config(args.config), "root")
    expected_top_level = {"model", "dataset", "trainer"}
    extra_top_level = set(config) - expected_top_level
    if extra_top_level:
        raise SystemExit(
            f"Unexpected top-level keys: {sorted(extra_top_level)}. "
            f"Expected only: {sorted(expected_top_level)}"
        )
    try:
        model_cfg = _require_dict(config["model"], "model")
        dataset_cfg = _require_dict(config["dataset"], "dataset")
        trainer_cfg = _require_dict(config["trainer"], "trainer")
    except KeyError as exc:
        raise SystemExit("Config must contain top-level sections: model, dataset, trainer") from exc

    try:
        model_name = model_cfg["name"]
        model_config = _require_dict(model_cfg["config"], "model.config")
        dataset_name = dataset_cfg["name"]
        dataset_config = _require_dict(dataset_cfg["config"], "dataset.config")
        trainer_name = trainer_cfg["name"]
        trainer_args = _require_dict(trainer_cfg["args"], "trainer.args")
    except KeyError as exc:
        raise SystemExit(f"Missing required config key: {exc}") from exc

    # --- Attention backend (must be set before model import/build) ---
    # Options: auto | sdpa | flash_attn2 | flash_attn3
    # Canonical config location: trainer.args.attention_backend
    attn_backend = trainer_args.get("attention_backend")
    if attn_backend:
        set_attention_backend(attn_backend)
        logger.info(f"Attention backend set to: {attn_backend}")

    model = get_model_builder(model_name)(model_config)
    dataset, processor = get_dataset_builder(dataset_name)(dataset_config)
    trainer = get_trainer_builder(trainer_name)(
        model=model,
        dataset=dataset,
        processor=processor,
        trainer_args=trainer_args,
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving final model...")
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
