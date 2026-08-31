#!/usr/bin/env python3
"""Evaluate a preregistered bounded V16.6 residual over frozen V15.7 outputs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_strict_track2_autoregressive_candidate import Accumulator, gains, metric_rows


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def frames(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value.copy()).to(device).permute(0, 1, 4, 2, 3).float().div(255)


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def lowpass_residual(residual: torch.Tensor, sigma: float) -> torch.Tensor:
    # residual is [N,T,C,H,W]; filter channels independently without seams.
    n, t, c, h, w = residual.shape
    kernel = gaussian_kernel(sigma, residual.device).expand(c, 1, -1, -1)
    radius = kernel.shape[-1] // 2
    flat = residual.flatten(0, 1)
    filtered = F.conv2d(F.pad(flat, (radius,) * 4, mode="replicate"), kernel, groups=c)
    return filtered.unflatten(0, (n, t))


def measurements(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    arm_right: torch.Tensor,
    success: torch.Tensor,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    before, after = Accumulator(), Accumulator()
    base_rows = metric_rows(baseline, target, context)
    candidate_rows = metric_rows(candidate, target, context)
    masks = {
        "overall": torch.ones_like(arm_right, dtype=torch.bool),
        "left": ~arm_right,
        "right": arm_right,
        "capture_success": success,
        "capture_failure": ~success,
    }
    for name, mask in masks.items():
        before.add(name, base_rows, mask)
        after.add(name, candidate_rows, mask)
    baseline_result, candidate_result = before.result(), after.result()
    return baseline_result, candidate_result, gains(baseline_result, candidate_result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.candidate_cache_dir.exists():
        raise SystemExit("refusing to overwrite V16.7 diagnostic outputs")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v167-bounded-student-residual-diagnostic-preregistration-v1":
        raise SystemExit("unexpected V16.7 preregistration format")
    with np.load(args.cache, allow_pickle=False) as values:
        cached = {name: values[name].copy() for name in values.files}
    required = {"context_last", "target", "baseline", "candidate", "arm_right", "capture_success"}
    if not required.issubset(cached):
        raise SystemExit(f"cache missing arrays: {sorted(required.difference(cached))}")

    device = torch.device(args.device)
    baseline = frames(cached["baseline"], device)
    student = frames(cached["candidate"], device)
    target = frames(cached["target"], device)
    context = torch.from_numpy(cached["context_last"].copy()).to(device).permute(0, 3, 1, 2).float().div(255)
    arm_right = torch.from_numpy(cached["arm_right"].copy()).to(device).bool()
    success = torch.from_numpy(cached["capture_success"].copy()).to(device).bool()
    residual = (student - baseline) * 255
    smoothed = {
        float(sigma): lowpass_residual(residual, float(sigma))
        for sigma in prereg["grid"]["gaussian_sigma_pixels"]
    }
    gates = prereg["development_admission"]
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for onset, clip, strength, sigma in itertools.product(
            prereg["grid"]["right_onset_zero_based"],
            prereg["grid"]["residual_clip_uint8"],
            prereg["grid"]["residual_strength"],
            prereg["grid"]["gaussian_sigma_pixels"],
        ):
            correction = smoothed[float(sigma)].clamp(-float(clip), float(clip)).mul(float(strength)).div(255)
            active = torch.zeros_like(correction[:, :, :1])
            active[arm_right, int(onset):] = 1
            composed = (baseline + active * correction).clamp(0, 1)
            # Runtime emits uint8, so selection must include quantization.
            composed = composed.mul(255).round().div(255)
            _, _, improvement = measurements(
                baseline, composed, target, context, arm_right, success
            )
            left_exact = bool(torch.equal(composed[~arm_right], baseline[~arm_right]))
            checks = {
                "left_prediction_bit_exact": left_exact,
                "overall_rgb": improvement["overall"]["rgb_mae_improvement_percent"] >= gates["overall_rgb_improvement_percent_min"],
                "right_rgb": improvement["right"]["rgb_mae_improvement_percent"] >= gates["right_rgb_improvement_percent_min"],
                "right_contact": improvement["right"]["contact_rgb_mae_improvement_percent"] >= gates["right_contact_improvement_percent_min"],
                "overall_texture": improvement["overall"]["texture_mae_improvement_percent"] >= gates["overall_texture_nonregression_tolerance_percent"],
                "overall_temporal": improvement["overall"]["temporal_delta_mae_improvement_percent"] >= gates["overall_temporal_nonregression_tolerance_percent"],
                "capture_success": improvement["capture_success"]["rgb_mae_improvement_percent"] >= gates["capture_success_nonregression_tolerance_percent"],
            }
            normalized_margins = [
                improvement["overall"]["rgb_mae_improvement_percent"] / gates["overall_rgb_improvement_percent_min"],
                improvement["right"]["rgb_mae_improvement_percent"] / gates["right_rgb_improvement_percent_min"],
                improvement["right"]["contact_rgb_mae_improvement_percent"] / gates["right_contact_improvement_percent_min"],
            ]
            rows.append({
                "onset": int(onset), "clip": float(clip), "strength": float(strength), "sigma": float(sigma),
                "passed": all(checks.values()), "checks": checks,
                "safety_margin": float(min(normalized_margins)),
                "improvement": improvement,
            })
    ranked = sorted(
        (row for row in rows if row["passed"]),
        key=lambda row: (row["safety_margin"], row["improvement"]["overall"]["rgb_mae_improvement_percent"]),
        reverse=True,
    )
    args.candidate_cache_dir.mkdir(parents=True)
    emitted = []
    for rank, row in enumerate(ranked[:5], 1):
        correction = smoothed[row["sigma"]].clamp(-row["clip"], row["clip"]).mul(row["strength"]).div(255)
        active = torch.zeros_like(correction[:, :, :1])
        active[arm_right, row["onset"]:] = 1
        prediction = (baseline + active * correction).clamp(0, 1).mul(255).round().byte()
        prediction = prediction.permute(0, 1, 3, 4, 2).cpu().numpy()
        path = args.candidate_cache_dir / f"rank{rank}.npz"
        np.savez_compressed(
            path,
            context_last=cached["context_last"], target=cached["target"],
            baseline=cached["baseline"], candidate=prediction,
            arm_right=cached["arm_right"], capture_success=cached["capture_success"],
            path=cached.get("path"), synthetic_seed=cached.get("synthetic_seed"), start=cached.get("start"),
        )
        emitted.append({"rank": rank, "cache": str(path.resolve()), "sha256": digest(path), **row})
    report = {
        "format": "strict-track2-v167-bounded-student-residual-grid-v1",
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": digest(args.preregistration),
        "source_cache": str(args.cache.resolve()),
        "source_cache_sha256": digest(args.cache),
        "grid_candidates": len(rows),
        "passed_candidates": len(ranked),
        "emitted": emitted,
        "candidates": rows,
        "interpretation_guard": "Development-cache evidence only; no parent or Track 2 promotion claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("grid_candidates", "passed_candidates", "emitted")}, indent=2))


if __name__ == "__main__":
    main()
