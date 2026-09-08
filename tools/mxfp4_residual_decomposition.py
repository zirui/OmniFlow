#!/usr/bin/env python3
"""Decompose MXFP4 Linear output error into activation and weight contributions."""

import argparse
import json
from pathlib import Path

import torch
from mxfp4_real_tensor_gate import evaluate_tensor


def metrics(value, reference):
    error = value.float() - reference.float()
    ref_norm = torch.linalg.vector_norm(reference.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "relative_l2": (torch.linalg.vector_norm(error) / ref_norm).item(),
        "mse": error.square().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policies", default="uos,current")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    for path in args.captures:
        capture = torch.load(path, map_location="cpu", weights_only=True)
        x = capture["input"][: args.rows].to(args.device).bfloat16()
        weight = capture["weight"][: args.out_features].to(args.device).bfloat16()
        d_exact = (x @ weight.T).float()

        for policy in args.policies.split(","):
            qx, _ = evaluate_tensor(x, policy, False)
            qw, _ = evaluate_tensor(weight, policy, True)
            qx = qx.bfloat16()
            qw = qw.bfloat16()

            a_quantized = (qx @ qw.T).float()
            b_activation_residual = a_quantized + ((x - qx) @ weight.T).float()
            c_weight_residual = a_quantized + (qx @ (weight - qw).T).float()

            a_metrics = metrics(a_quantized, d_exact)
            b_metrics = metrics(b_activation_residual, d_exact)
            c_metrics = metrics(c_weight_residual, d_exact)
            mse_a = max(a_metrics["mse"], torch.finfo(torch.float32).tiny)
            results.append(
                {
                    "file": path.name,
                    "step": capture["step"],
                    "module": capture["module"],
                    "policy": policy,
                    "A_QX_QW": a_metrics,
                    "B_activation_residual": b_metrics,
                    "C_weight_residual": c_metrics,
                    "D_exact_bf16": {"relative_l2": 0.0, "mse": 0.0},
                    "remaining_weight_error_fraction_after_B": b_metrics["mse"] / mse_a,
                    "remaining_activation_error_fraction_after_C": c_metrics["mse"] / mse_a,
                    "activation_correction_error_reduction": 1.0 - b_metrics["mse"] / mse_a,
                    "weight_correction_error_reduction": 1.0 - c_metrics["mse"] / mse_a,
                }
            )

    text = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
