#!/usr/bin/env python3
"""Compare legal E8M0 scale policies on captured FLUX Linear tensors."""

import argparse
import json
import time
from pathlib import Path

import torch

LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def to_blocks(x, block_2d=False):
    rows, cols = x.shape
    pad_rows = ((rows + 31) // 32) * 32 if block_2d else rows
    pad_cols = ((cols + 31) // 32) * 32
    padded = torch.zeros(pad_rows, pad_cols, device=x.device, dtype=torch.float32)
    padded[:rows, :cols] = x.float()
    if block_2d:
        blocks = padded.view(pad_rows // 32, 32, pad_cols // 32, 32)
        blocks = blocks.permute(0, 2, 1, 3).reshape(-1, 1024)
    else:
        blocks = padded.view(-1, 32)
    return blocks, (rows, cols, pad_rows, pad_cols)


def from_blocks(blocks, shape, block_2d=False):
    rows, cols, pad_rows, pad_cols = shape
    if block_2d:
        x = blocks.view(pad_rows // 32, pad_cols // 32, 32, 32)
        x = x.permute(0, 2, 1, 3).reshape(pad_rows, pad_cols)
    else:
        x = blocks.view(pad_rows, pad_cols)
    return x[:rows, :cols]


def round_fp4(x):
    levels = LEVELS.to(x.device)
    bounds = BOUNDS.to(x.device)
    return torch.sign(x) * levels[torch.bucketize(x.abs(), bounds)]


def threshold_scale(blocks, threshold):
    amax = blocks.abs().amax(dim=1, keepdim=True)
    exponent = torch.floor(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny)))
    mantissa = amax / torch.exp2(exponent)
    return torch.exp2(exponent - 2 + (mantissa >= threshold))


def ideal_scale(blocks):
    return blocks.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny) / 6.0


def quantize(blocks, scale):
    return round_fp4(blocks / scale).clamp(-6, 6) * scale


def scalesearch(blocks):
    amax = blocks.abs().amax(dim=1, keepdim=True)
    base = torch.floor(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny))) - 2
    candidates = torch.stack([torch.exp2(base + offset) for offset in (-1, 0, 1)], dim=0)
    errors = torch.stack([((quantize(blocks, scale) - blocks) ** 2).mean(dim=1) for scale in candidates])
    best = errors.argmin(dim=0)
    return candidates[best, torch.arange(blocks.shape[0], device=blocks.device)]


def evaluate_tensor(x, policy, block_2d):
    blocks, shape = to_blocks(x, block_2d)
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    if policy == "scalesearch":
        scale = scalesearch(blocks)
    else:
        threshold = {"current": 1.75, "uos": 1.8125}[policy]
        scale = threshold_scale(blocks, threshold)
    q = quantize(blocks, scale)
    if x.is_cuda:
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    ideal = quantize(blocks, ideal_scale(blocks))
    deadzone = blocks.abs() < blocks.abs().amax(dim=1, keepdim=True) / 24.0
    clipped = (blocks / scale).abs() > 6.0
    total_energy = blocks.square().sum().clamp_min(torch.finfo(torch.float32).tiny)
    metrics = {
        "relative_l2": (torch.linalg.vector_norm(q - blocks) / torch.linalg.vector_norm(blocks)).item(),
        "scale_bias_energy": ((q - ideal).square().sum() / total_energy).item(),
        "deadzone_ratio": deadzone.float().mean().item(),
        "deadzone_energy": (blocks.square()[deadzone].sum() / total_energy).item(),
        "grid_noise_energy": ((ideal - blocks).square()[~deadzone].sum() / total_energy).item(),
        "clipping_ratio": clipped.float().mean().item(),
        "clipping_energy": (blocks.square()[clipped].sum() / total_energy).item(),
        "reference_quantizer_ms": elapsed_ms,
    }
    return from_blocks(q, shape, block_2d), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = []
    for path in args.captures:
        capture = torch.load(path, map_location="cpu", weights_only=True)
        x = capture["input"][: args.rows].to(args.device)
        weight = capture["weight"][: args.out_features].to(args.device)
        reference = x.float() @ weight.float().T
        for policy in ("current", "uos", "scalesearch"):
            qx, x_metrics = evaluate_tensor(x, policy, False)
            qw, w_metrics = evaluate_tensor(weight, policy, True)
            output = qx @ qw.T
            output_error = torch.linalg.vector_norm(output - reference) / torch.linalg.vector_norm(reference)
            results.append(
                {
                    "file": path.name,
                    "step": capture["step"],
                    "module": capture["module"],
                    "policy": policy,
                    "gemm_output_relative_l2": output_error.item(),
                    "activation": x_metrics,
                    "weight": w_metrics,
                }
            )
    text = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
