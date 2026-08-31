#!/usr/bin/env python3
"""Cache continuous v23.4 parent predictions for video export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wam_pipeline.action_occlusion_texture_parent_v230 import ActionOcclusionTextureParentV230
from train_action_occlusion_texture_parent_v230 import (
    load_rgb_parent,
    recurrent_rollout,
    rgb_parent_rollout,
)
from train_autoregressive_unet import WindowDataset, frames_for_model
from train_trajectory_motion_renderer_v220 import load_pose, project_future


def rgb8(value):
    return value.mul(255).round().clamp(0, 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", type=int)
    selection.add_argument(
        "--selection-cache",
        help="NPZ containing the exact ordered `windows` field to evaluate",
    )
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--extension-checkpoint", required=True)
    parser.add_argument("--residual-ema", type=float, default=.6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.selection_cache:
        with np.load(args.selection_cache, allow_pickle=False) as selection_cache:
            selected_names = selection_cache["windows"].astype(str).tolist()
        episodes = sorted({int(name.split("_")[0][7:]) for name in selected_names})
        data = WindowDataset(Path(args.windows), episodes)
        name_to_index = {path.name: index for index, path in enumerate(data.paths)}
        missing = [name for name in selected_names if name not in name_to_index]
        if missing:
            raise FileNotFoundError(f"selection contains missing windows: {missing[:5]}")
        evaluation_data = Subset(data, [name_to_index[name] for name in selected_names])
        output_stem = "validation64"
    else:
        data = WindowDataset(Path(args.windows), [args.episode])
        selected_names = [path.name for path in data.paths]
        evaluation_data = data
        output_stem = f"episode{args.episode}"
    loader = DataLoader(evaluation_data, batch_size=1, num_workers=2, pin_memory=True)
    parent, mean, std = load_rgb_parent(Path(args.init_checkpoint), device)
    pose_model, pose_stats = load_pose(args.pose_checkpoint, device)
    checkpoint = Path(args.extension_checkpoint)
    state = torch.load(checkpoint / "parent_extension.pt", map_location="cpu", weights_only=True)
    extension = ActionOcclusionTextureParentV230(**state.get("config", {})).to(device)
    extension.load_state_dict(state["state_dict"], strict=True)
    extension.eval().requires_grad_(False)
    parents, predictions, targets, contexts, arm_ids = [], [], [], [], []
    for index, (raw_context, raw_history, raw_future, raw_target) in enumerate(loader):
        raw_history = raw_history.to(device)
        raw_future = raw_future.to(device)
        context = frames_for_model(raw_context).to(device)
        poses, arms = project_future(pose_model, pose_stats, raw_history, raw_future)
        history = ((raw_history - mean) / std).float()
        future = ((raw_future - mean) / std).float()
        base = rgb_parent_rollout(parent, context.clone(), history.clone(), future)
        value, _, _ = recurrent_rollout(
            parent, extension, context, history, future, context.clone(), poses, arms,
            state_strength=0.0, residual_ema=args.residual_ema,
        )
        parents.append(rgb8(base)[0])
        predictions.append(rgb8(value)[0])
        targets.append(raw_target.numpy()[0])
        contexts.append(raw_context.numpy()[0])
        arm_ids.append(int(arms[0]))
        if (index + 1) % 50 == 0:
            print(json.dumps({"processed": index + 1, "total": len(evaluation_data)}), flush=True)
    parent_array = np.stack(parents)
    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)
    context_array = np.stack(contexts)
    names = np.asarray(selected_names)
    arm_array = np.asarray(arm_ids, dtype=np.int64)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{output_stem}_parent.npz",
        prediction=parent_array, target=target_array, context=context_array,
        windows=names, arm_id=arm_array,
    )
    np.savez_compressed(
        output / f"{output_stem}_v234.npz",
        prediction=prediction_array, target=target_array, context=context_array,
        windows=names, arm_id=arm_array,
    )
    metrics = {
        "format": "v23.4-continuous-cache",
        "episode": args.episode,
        "selection_cache": args.selection_cache,
        "windows": len(evaluation_data),
        "residual_ema": args.residual_ema,
        "parent_rgb_mae": float(np.abs(parent_array.astype(np.float32) - target_array).mean()),
        "v234_rgb_mae": float(np.abs(prediction_array.astype(np.float32) - target_array).mean()),
    }
    (output / f"{output_stem}_cache_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
