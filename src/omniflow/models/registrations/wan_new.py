"""Register Wan (new) model builder."""

import torch
from loguru import logger

from omniflow.models.wan_new import WanVideoForConditionalGeneration as WanVideoForConditionalGenerationNew
from omniflow.models.wan_new.configuration_wanvideo import WanVideoConfig as WanVideoConfigNew
from omniflow.registry import register_model
from omniflow.utils.train_utils import count_parameters


def load_encoder_weights(model, config):
    if "encoder" not in config:
        return

    encoder_cfg = config["encoder"]

    if "t5_encoder" in encoder_cfg:
        t5_path = encoder_cfg["t5_encoder"]
        logger.info(f"Loading T5 Encoder from {t5_path}")
        try:
            state_dict = torch.load(t5_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

            keys_result = model.text_encoder.load_state_dict(state_dict, strict=False)
            logger.info(
                f"T5 Encoder loaded. Missing keys: {len(keys_result.missing_keys)}, "
                f"Unexpected keys: {len(keys_result.unexpected_keys)}"
            )
            model.text_encoder.to(torch.bfloat16)
            logger.info("T5 loaded and cast to bfloat16.")
        except Exception as exc:
            logger.error(f"Failed to load T5 Encoder: {exc}")

    if "autoencoder" in encoder_cfg:
        vae_path = encoder_cfg["autoencoder"]
        logger.info(f"Loading VAE from {vae_path}")
        try:
            state_dict = torch.load(vae_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            keys_result = model.vae.load_state_dict(state_dict, strict=False)
            logger.info(
                f"VAE loaded. Missing keys: {len(keys_result.missing_keys)}, "
                f"Unexpected keys: {len(keys_result.unexpected_keys)}"
            )
            model.vae.to(torch.bfloat16)
            logger.info("VAE loaded and cast to bfloat16.")
        except Exception as exc:
            logger.error(f"Failed to load VAE: {exc}")


@register_model("wan_new")
def build_wan_new_model(model_config: dict):
    encoder_cfg = model_config.get("encoder", {}) if isinstance(model_config.get("encoder"), dict) else {}
    config_dict = dict(model_config.get("config", {}))
    if encoder_cfg:
        config_dict.setdefault("encoder", encoder_cfg)

    if "load_from_pretrained_path" in model_config:
        pretrained_path = model_config["load_from_pretrained_path"]
        if "config" in model_config and hasattr(WanVideoForConditionalGenerationNew, "load_dit"):
            logger.info(f"Loading DiT model from {pretrained_path} with config overrides")
            model = WanVideoForConditionalGenerationNew.load_dit(pretrained_path, **config_dict)
        else:
            try:
                if hasattr(WanVideoForConditionalGenerationNew, "load_dit"):
                    logger.info(f"Attempting to load as DiT checkpoint from {pretrained_path}")
                    model = WanVideoForConditionalGenerationNew.load_dit(pretrained_path)
                else:
                    raise AttributeError("load_dit is not available")
            except Exception as exc:
                logger.warning(f"load_dit failed: {exc}. Falling back to from_pretrained.")
                model = WanVideoForConditionalGenerationNew.from_pretrained(pretrained_path)

        if "trainable_modules" in model_config:
            if hasattr(model, "config"):
                model.config.trainable_modules = model_config["trainable_modules"]
            model.trainable_modules = model_config["trainable_modules"]
    elif "config" in model_config:
        logger.info("Building model from scratch")
        model_cfg = WanVideoConfigNew(**config_dict)
        model = WanVideoForConditionalGenerationNew(model_cfg)
    else:
        raise ValueError("model_config must contain either 'load_from_pretrained_path' or 'config'")

    total_params, trainable_params = count_parameters(model)
    logger.info(
        f"parameters: total_params={total_params/1e9:.2f}B, "
        f"trainable_params={trainable_params/1e9:.2f}B"
    )

    if hasattr(model, "freeze_except"):
        model.freeze_except()
        total_params, trainable_params = count_parameters(model)
        logger.info(
            f"parameters after freezing: total_params={total_params/1e9:.2f}B, "
            f"trainable_params={trainable_params/1e9:.2f}B"
        )

    if hasattr(model, "load_encoder_weights"):
        model.load_encoder_weights(encoder_cfg)
    else:
        load_encoder_weights(model, model_config)
    total_params, trainable_params = count_parameters(model)
    logger.info(
        f"parameters after loading encoder weights: total_params={total_params/1e9:.2f}B, "
        f"trainable_params={trainable_params/1e9:.2f}B"
    )
    return model
