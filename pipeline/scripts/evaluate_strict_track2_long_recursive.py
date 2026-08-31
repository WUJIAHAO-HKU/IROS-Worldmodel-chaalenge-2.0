#!/usr/bin/env python3
"""True long-horizon recursive evaluation for Track-2 autoregressive parents.

Each held-out episode is initialized from its first five ground-truth frames.
After that, the model consumes only its own predicted frames and the recorded
actions.  Targets are used solely after inference to compute metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wam_pipeline.autoregressive_unet import OneStepActionUNet


def load_model(path: Path, device: torch.device):
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError(f"unsupported checkpoint: {path}")
    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    with np.load(path / "action_normalization.npz", allow_pickle=False) as values:
        mean = torch.from_numpy(values["mean"].astype(np.float32)).to(device)
        std = torch.from_numpy(values["std"].astype(np.float32)).to(device)
    return model, mean, std


def start_id(path: Path) -> int:
    match = re.search(r"_(\d+)\.npz$", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def episode_inputs(root: Path, episode: int, max_steps: int):
    paths = sorted(root.glob(f"episode{episode}_*.npz"), key=start_id)
    if not paths:
        raise RuntimeError(f"no windows for episode {episode}")
    first_start = start_id(paths[0])
    with np.load(paths[0], allow_pickle=False) as first:
        context = first["context_frames"].copy()
        history = first["history_actions"].astype(np.float32).copy()
    # Official windows use stride 1 while captured on-policy windows may use
    # stride 4.  Reconstruct the dense timeline from every window's eight
    # future entries; overlapping entries are identical and the earliest copy
    # is retained deterministically.
    timeline: dict[int, tuple[np.ndarray, np.ndarray, bool, bool]] = {}
    for path in paths:
        start = start_id(path)
        with np.load(path, allow_pickle=False) as data:
            future = data["future_actions"].astype(np.float32)
            target = data["target_frames"]
            arm_right = bool(data["arm_right"]) if "arm_right" in data.files else False
            capture_success = bool(data["capture_success"]) if "capture_success" in data.files else True
            for offset in range(min(len(future), len(target))):
                timeline.setdefault(
                    start + offset,
                    (future[offset].copy(), target[offset].copy(), arm_right, capture_success),
                )
    actions, targets, arm_labels, success_labels = [], [], [], []
    for index in range(first_start, first_start + max_steps):
        if index not in timeline:
            break
        action, target, arm_right, capture_success = timeline[index]
        actions.append(action)
        targets.append(target)
        arm_labels.append(arm_right)
        success_labels.append(capture_success)
    if len(actions) < 8:
        raise RuntimeError(f"episode {episode} has only {len(actions)} reconstructed steps")
    return (context, history, np.stack(actions), np.stack(targets),
            np.asarray(arm_labels, dtype=bool), np.asarray(success_labels, dtype=bool))


def spatial_edge(value: torch.Tensor) -> torch.Tensor:
    dx = (value[..., :, 1:] - value[..., :, :-1]).abs().mean((-3, -2, -1))
    dy = (value[..., 1:, :] - value[..., :-1, :]).abs().mean((-3, -2, -1))
    return 0.5 * (dx + dy)


def highpass(value: torch.Tensor) -> torch.Tensor:
    return value - F.avg_pool2d(value, 3, 1, 1, count_include_pad=False)


def infer(model, mean, std, context_np, history_np, actions_np, device, lowfreq_alpha: float = 0.0, residual_clip: float = 0.0):
    context = torch.from_numpy(context_np).permute(0, 3, 1, 2).to(device).float().div(255)[None]
    raw_history = torch.from_numpy(history_np).to(device)[None]
    predictions = []
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for raw_action in torch.from_numpy(actions_np).to(device).unbind(0):
            action_sequence = torch.cat((raw_history, raw_action[None, None]), 1)
            raw_prediction = model(context, (action_sequence - mean) / std)
            if residual_clip > 0.0:
                previous = context[:, -1]
                prediction = (previous + (raw_prediction - previous).clamp(-residual_clip, residual_clip)).clamp(0, 1)
            else:
                prediction = raw_prediction.clamp(0, 1)
            if lowfreq_alpha > 0.0:
                previous = context[:, -1]
                prediction_low = F.avg_pool2d(prediction, 5, 1, 2)
                previous_low = F.avg_pool2d(previous, 5, 1, 2)
                prediction = (prediction + lowfreq_alpha * (previous_low - prediction_low)).clamp(0, 1)
            predictions.append(prediction.float().cpu())
            context = torch.cat((context[:, 1:], prediction[:, None]), 1)
            raw_history = torch.cat((raw_history[:, 1:], raw_action[None, None]), 1)
    return torch.cat(predictions, 0)


def metrics(prediction: torch.Tensor, target_np: np.ndarray, context_np: np.ndarray) -> dict:
    target = torch.from_numpy(target_np).permute(0, 3, 1, 2).float().div(255)
    previous_prediction = torch.cat((
        torch.from_numpy(context_np[-1]).permute(2, 0, 1).float().div(255)[None],
        prediction[:-1],
    ))
    previous_target = torch.cat((
        torch.from_numpy(context_np[-1]).permute(2, 0, 1).float().div(255)[None],
        target[:-1],
    ))
    mae = 255 * (prediction - target).abs().mean((1, 2, 3))
    texture = 255 * (highpass(prediction) - highpass(target)).abs().mean((1, 2, 3))
    temporal = 255 * ((prediction - previous_prediction) - (target - previous_target)).abs().mean((1, 2, 3))
    edge_ratio = spatial_edge(prediction) / spatial_edge(target).clamp_min(1e-6)
    contrast_ratio = prediction.std((1, 2, 3), unbiased=False) / target.std((1, 2, 3), unbiased=False).clamp_min(1e-6)
    return {
        "rgb_mae": mae.numpy(), "texture_mae": texture.numpy(),
        "temporal_delta_mae": temporal.numpy(), "edge_energy_ratio": edge_ratio.numpy(),
        "contrast_ratio": contrast_ratio.numpy(),
    }


def reduce(rows: list[dict]) -> dict:
    output = {"episodes": len(rows), "steps": sum(len(row["rgb_mae"]) for row in rows)}
    for key in rows[0]:
        arrays = [row[key] for row in rows]
        flat = np.concatenate(arrays)
        late = np.concatenate([array[63:] for array in arrays if len(array) > 63])
        output[key] = float(flat.mean())
        output[f"late64_{key}"] = float(late.mean()) if len(late) else None
        horizon = {}
        for position in (1, 8, 16, 32, 64, 128, 192, 200):
            values = [float(array[position - 1]) for array in arrays if len(array) >= position]
            if values:
                horizon[str(position)] = float(np.mean(values))
        output[f"{key}_by_horizon"] = horizon
    return output


def summarize(records: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        values = record[key]
        masks = {
            "overall": np.ones(len(record["arm_right"]), dtype=bool),
            "left": ~record["arm_right"], "right": record["arm_right"],
            "capture_success": record["capture_success"],
            "capture_failure": ~record["capture_success"],
        }
        for group, mask in masks.items():
            if mask.any():
                groups[group].append({name: array[mask] for name, array in values.items()})
    return {group: reduce(rows) for group, rows in groups.items()}


def improvement(reference: dict, candidate: dict) -> dict:
    result = {}
    for group in sorted(reference.keys() & candidate.keys()):
        result[group] = {}
        for key in ("rgb_mae", "texture_mae", "temporal_delta_mae", "late64_rgb_mae",
                    "late64_texture_mae", "late64_temporal_delta_mae"):
            before, after = reference[group].get(key), candidate[group].get(key)
            if before is not None and after is not None:
                result[group][f"{key}_percent"] = 100 * (before - after) / before
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-right", type=Path,
                        help="optional instruction-routed right-arm checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-lowfreq-alpha", type=float, default=0.0)
    parser.add_argument("--candidate-residual-clip", type=float, default=0.0)
    args = parser.parse_args()
    episodes = json.loads(args.split_manifest.read_text())["validation_episodes"]
    device = torch.device(args.device)
    baseline_model, baseline_mean, baseline_std = load_model(args.baseline, device)
    candidate_model, candidate_mean, candidate_std = load_model(args.candidate, device)
    if args.candidate_right:
        right_model, right_mean, right_std = load_model(args.candidate_right, device)
    else:
        right_model, right_mean, right_std = candidate_model, candidate_mean, candidate_std
    records = []
    for episode in episodes:
        context, history, actions, targets, arm_right, success = episode_inputs(
            args.windows, episode, args.max_steps
        )
        before = metrics(infer(baseline_model, baseline_mean, baseline_std, context, history, actions, device), targets, context)
        use_right = bool(arm_right[0]) if len(arm_right) else False
        selected_model, selected_mean, selected_std = (
            (right_model, right_mean, right_std) if use_right
            else (candidate_model, candidate_mean, candidate_std)
        )
        after = metrics(infer(selected_model, selected_mean, selected_std, context, history, actions, device, args.candidate_lowfreq_alpha, args.candidate_residual_clip), targets, context)
        records.append({"episode": episode, "arm_right": arm_right, "capture_success": success,
                        "steps": len(actions), "baseline": before, "candidate": after})
        print(json.dumps({"episode": episode, "steps": len(actions),
                          "right_steps": int(arm_right.sum()),
                          "capture_success_steps": int(success.sum())}), flush=True)
    baseline = summarize(records, "baseline")
    candidate = summarize(records, "candidate")
    report = {
        "format": "strict-track2-true-recursive-heldout-v1",
        "inference": "five initial ground-truth frames, then predictions only; targets are metric-only",
        "episodes": episodes,
        "baseline_checkpoint": str(args.baseline.resolve()),
        "candidate_checkpoint": str(args.candidate.resolve()),
        "candidate_right_checkpoint": str(args.candidate_right.resolve()) if args.candidate_right else None,
        "candidate_lowfreq_alpha": args.candidate_lowfreq_alpha,
        "candidate_residual_clip": args.candidate_residual_clip,
        "baseline": baseline,
        "candidate": candidate,
        "improvement": improvement(baseline, candidate),
    }
    overall = report["improvement"]["overall"]
    arms = [report["improvement"][arm] for arm in ("left", "right") if arm in report["improvement"]]
    report["promotion_gate"] = {
        "pass": overall["rgb_mae_percent"] > 0 and overall["late64_rgb_mae_percent"] > 0
                and overall["texture_mae_percent"] > 0 and all(x["rgb_mae_percent"] >= 0 for x in arms),
        "rule": "overall RGB, late64 RGB and texture improve; neither arm RGB regresses",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["promotion_gate"]), flush=True)


if __name__ == "__main__":
    main()
