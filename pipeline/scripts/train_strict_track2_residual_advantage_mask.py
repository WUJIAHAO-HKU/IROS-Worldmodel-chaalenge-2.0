#!/usr/bin/env python3
"""Learn target-free deployment masks from training-only visual advantage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_post_v15_residual import load_checkpoint
from train_strict_track2_post_v15_residual import CachedSequences, active_actions, frames, rollout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.55,0.6")
    parser.add_argument("--hybrid-official-left-thresholds", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    hybrid_thresholds = [
        float(value) for value in args.hybrid_official_left_thresholds.split(",") if value
    ]
    if not thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("thresholds must be in (0, 1]")

    dataset = CachedSequences(Path(args.cache))
    device = torch.device(args.device)
    model, mean, std, config = load_checkpoint(Path(args.checkpoint), device)
    advantage_sum = np.zeros((2, 2, 8, 256, 256), dtype=np.float64)
    beneficial_count = np.zeros((2, 2, 8, 256, 256), dtype=np.uint16)
    stratum_count = np.zeros((2, 2), dtype=np.int64)
    with torch.inference_mode():
        for index in range(len(dataset)):
            raw_context, raw_future, raw_baseline, raw_target, arm_right, _ = dataset[index]
            context = frames(raw_context[None, None], device)[:, 0]
            baseline = frames(raw_baseline[None], device)
            target = frames(raw_target[None], device)
            future = raw_future[None].to(device)
            arm_tensor = arm_right[None].to(device).bool()
            actions = active_actions(future, arm_tensor)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, _ = rollout(model, baseline, context, actions, arm_tensor, mean, std)
            deployed = prediction.float().mul(255).round().clamp(0, 255)
            baseline_255 = baseline.mul(255)
            target_255 = target.mul(255)
            advantage = (
                (baseline_255 - target_255).abs() - (deployed - target_255).abs()
            ).mean(2)[0].cpu().numpy()
            source = int(dataset.is_synthetic[index])
            arm = int(dataset.arm_right[index])
            advantage_sum[source, arm] += advantage
            beneficial_count[source, arm] += (advantage > 0).astype(np.uint16)
            stratum_count[source, arm] += 1
            if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(dataset):
                print(json.dumps({"processed": index + 1, "total": len(dataset)}), flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    denominator = stratum_count[:, :, None, None, None]
    mean_advantage = advantage_sum / denominator
    fraction = beneficial_count.astype(np.float64) / denominator
    masks = {}
    for threshold in thresholds:
        mask = (mean_advantage > 0) & (fraction >= threshold)
        masks[threshold] = mask
        tag = f"{round(100 * threshold):03d}"
        path = output / f"advantage_mask_fraction_{tag}.npz"
        np.savez_compressed(
            path,
            format=np.asarray("strict-track2-residual-advantage-mask-v1"),
            mask=mask.astype(np.uint8),
            beneficial_fraction_threshold=np.asarray(threshold, dtype=np.float32),
            stratum_count=stratum_count,
            coverage=mask.mean(axis=(2, 3, 4)).astype(np.float32),
            checkpoint_config=np.asarray(json.dumps(config, sort_keys=True)),
        )
        records.append({
            "threshold": threshold,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "coverage_by_source_arm": mask.mean(axis=(2, 3, 4)).tolist(),
        })
    if hybrid_thresholds:
        base_threshold = thresholds[0]
        for threshold in hybrid_thresholds:
            strict = (mean_advantage > 0) & (fraction >= threshold)
            mask = masks[base_threshold].copy()
            mask[0, 0] = strict[0, 0]
            tag = f"hybrid_base{round(100 * base_threshold):03d}_official_left{round(100 * threshold):03d}"
            path = output / f"advantage_mask_{tag}.npz"
            np.savez_compressed(
                path,
                format=np.asarray("strict-track2-residual-advantage-mask-v1"),
                mask=mask.astype(np.uint8),
                beneficial_fraction_threshold=np.asarray(base_threshold, dtype=np.float32),
                official_left_beneficial_fraction_threshold=np.asarray(threshold, dtype=np.float32),
                stratum_count=stratum_count,
                coverage=mask.mean(axis=(2, 3, 4)).astype(np.float32),
                checkpoint_config=np.asarray(json.dumps(config, sort_keys=True)),
            )
            records.append({
                "threshold": base_threshold,
                "official_left_threshold": threshold,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "coverage_by_source_arm": mask.mean(axis=(2, 3, 4)).tolist(),
            })
    manifest = {
        "format": "strict-track2-residual-advantage-mask-training-v1",
        "cache": str(Path(args.cache).resolve()),
        "cache_manifest_sha256": sha256(Path(args.cache) / "manifest.json"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(Path(args.checkpoint) / "post_v15_residual.pt"),
        "source_gate": str(Path(args.source_gate).resolve()),
        "source_gate_sha256": sha256(Path(args.source_gate)),
        "preregistration": str(Path(args.preregistration).resolve()),
        "preregistration_sha256": sha256(Path(args.preregistration)),
        "stratum_count": stratum_count.tolist(),
        "records": records,
        "inference_uses_target_or_reward": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
