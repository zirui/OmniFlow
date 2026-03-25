#!/usr/bin/env python3
"""Unified training entry point (registry-based CLI)."""

import argparse
import logging
import os

import yaml


def _is_primary_process() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _require_dict(obj: object, name: str) -> dict:
    if not isinstance(obj, dict):
        raise SystemExit(f"Config '{name}' must be a dict, got: {type(obj).__name__}")
    return obj


def _parse_override_value(raw_value: str) -> object:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise SystemExit(f"Failed to parse override value '{raw_value}': {exc}") from exc


def _parse_override(override: str) -> tuple[str, object]:
    if "=" not in override:
        raise SystemExit(
            "Override must use PATH=VALUE format, "
            "for example: trainer.args.per_device_train_batch_size=32"
        )
    path, raw_value = override.split("=", 1)
    path = path.strip()
    if not path:
        raise SystemExit("Override path cannot be empty.")
    if any(not part for part in path.split(".")):
        raise SystemExit(f"Override path contains an empty segment: {path}")
    return path, _parse_override_value(raw_value)


def apply_overrides(config: dict, overrides: list[str]) -> None:
    for override in overrides:
        path, value = _parse_override(override)
        parts = path.split(".")
        cursor = config
        cursor_name = "root"

        for part in parts[:-1]:
            if part not in cursor:
                raise SystemExit(f"Override path not found: {path} (missing '{part}')")
            cursor_name = f"{cursor_name}.{part}"
            cursor = _require_dict(cursor[part], cursor_name)

        leaf = parts[-1]
        if leaf not in cursor:
            raise SystemExit(f"Override path not found: {path} (missing '{leaf}')")
        cursor[leaf] = value


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("omniflow")
    is_primary_process = _is_primary_process()

    parser = argparse.ArgumentParser(description="OmniFlow Unified Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override a config value after loading YAML. "
            "Example: --set trainer.args.per_device_train_batch_size=32"
        ),
    )
    args = parser.parse_args(argv)

    from omniflow.attention import set_attention_backend
    from omniflow.registry import get_dataset_builder, get_model_builder, get_trainer_builder

    if is_primary_process:
        logger.info(f"Loading config from: {args.config}")
    config = _require_dict(load_config(args.config), "root")
    apply_overrides(config, args.overrides)
    if is_primary_process and args.overrides:
        logger.info("Applied config overrides:")
        for override in args.overrides:
            logger.info(f"  {override}")
    if is_primary_process:
        final_config = yaml.safe_dump(config, sort_keys=False).rstrip()
        logger.info(f"Final merged config:\n{final_config}")
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
    # Options: auto | sdpa | flex_attention | flash_attn2 | flash_attn3 | flash_attn_aiter
    # Canonical config location: trainer.args.attention_backend
    attn_backend = trainer_args.get("attention_backend")
    if attn_backend:
        set_attention_backend(attn_backend)
        if is_primary_process:
            logger.info(f"Attention backend set to: {attn_backend}")

    model = get_model_builder(model_name)(model_config)
    dataset, processor = get_dataset_builder(dataset_name)(dataset_config)
    trainer = get_trainer_builder(trainer_name)(
        model=model,
        dataset=dataset,
        processor=processor,
        trainer_args=trainer_args,
    )
    if is_primary_process:
        logger.info("Starting training...")
    trainer.train()
    if is_primary_process:
        logger.info("Saving final model...")
    trainer.save_model()
    if is_primary_process:
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
