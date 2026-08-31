#!/usr/bin/env python3
"""Full FP32 evaluation for recurrent v23 parent checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wam_pipeline.action_occlusion_texture_parent_v230 import (
    ActionOcclusionTextureParentV230,
    parameter_count,
)
from train_action_occlusion_texture_parent_v230 import evaluate, load_rgb_parent
from train_autoregressive_unet import WindowDataset
from train_trajectory_motion_renderer_v220 import load_pose


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--state-strengths", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.25, 1.0])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--active-until", nargs="+", type=int, default=[8])
    parser.add_argument("--residual-emas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    split = json.loads(Path(args.split_manifest).read_text())
    data = WindowDataset(Path(args.windows), split["validation_episodes"])
    evaluation_data = data
    if args.max_windows and args.max_windows < len(data):
        indices = np.linspace(0, len(data) - 1, args.max_windows, dtype=int).tolist()
        evaluation_data = Subset(data, indices)
    loader = DataLoader(evaluation_data, batch_size=1, num_workers=2, pin_memory=True)
    parent, mean, std = load_rgb_parent(Path(args.init_checkpoint), device)
    pose_model, pose_stats = load_pose(args.pose_checkpoint, device)
    results = []
    for value in args.checkpoints:
        checkpoint = Path(value)
        state_path = checkpoint / "parent_extension.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("format") != "track2-action-occlusion-texture-parent-v23.0":
            raise ValueError(f"unsupported v23 checkpoint: {checkpoint}")
        model = ActionOcclusionTextureParentV230(**state.get("config", {})).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        for strength in args.state_strengths:
          for active_until in args.active_until:
            for residual_ema in args.residual_emas:
                output_strengths = [1.0 if index < active_until else 0.0 for index in range(8)]
                metrics = evaluate(
                    loader, parent, pose_model, pose_stats, model, device, mean, std,
                    state_strength=strength, output_strengths=output_strengths,
                    residual_ema=residual_ema,
                )
                metrics.update({
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(state_path),
                    "state_strength": strength,
                    "active_until": active_until,
                    "residual_ema": residual_ema,
                })
                results.append(metrics)
                print(json.dumps(metrics), flush=True)
    results.sort(key=lambda item: item["gate_score"], reverse=True)
    report = {
        "format": "track2-v23.0-recurrent-parent-full-dev269",
        "data_boundary": "supplied_50_episodes_only",
        "episodes": split["validation_episodes"],
        "windows": len(evaluation_data),
        "frames": len(evaluation_data) * 8,
        "extension_parameters": parameter_count(),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
