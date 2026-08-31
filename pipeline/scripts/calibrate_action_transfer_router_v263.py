#!/usr/bin/env python3
"""Calibrate an action-only gate between the original and v26 action donor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_autoregressive_unet import WindowDataset, frames_for_model, rollout
from wam_pipeline.autoregressive_unet import OneStepActionUNet


def load_model(path: Path, device: torch.device) -> OneStepActionUNet:
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError(f"unsupported checkpoint: {path}")
    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.eval()


def improvement(base: np.ndarray, value: np.ndarray) -> float:
    return float((base.mean() - value.mean()) / base.mean() * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--min-active-horizon-improvement", type=float, default=0.05,
        help="Minimum dev improvement (percent) required at every horizon where the candidate is active.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    base_path, candidate_path = Path(args.base), Path(args.candidate)
    with np.load(base_path / "action_normalization.npz", allow_pickle=False) as data:
        mean = torch.from_numpy(np.asarray(data["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(data["std"], dtype=np.float32)).to(device)
    split = json.loads(Path(args.split).read_text())
    dataset = WindowDataset(Path(args.windows), split["validation_episodes"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    base_model, candidate_model = load_model(base_path, device), load_model(candidate_path, device)
    base_errors, candidate_errors, true_motion = [], [], []
    features = {"mean_velocity": [], "max_displacement": [], "end_displacement": []}
    arms = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for context, history, future, target in loader:
            context_model = frames_for_model(context).to(device)
            target_model = frames_for_model(target).to(device)
            history_device = history.to(device)
            future_device = future.to(device)
            normalized_history = ((history_device - mean) / std).float()
            normalized_future = ((future_device - mean) / std).float()
            base_prediction = rollout(base_model, context_model, normalized_history, normalized_future)
            candidate_prediction = rollout(candidate_model, context_model, normalized_history, normalized_future)
            base_errors.append((base_prediction.float() - target_model).abs().mean((2, 3, 4)).cpu().numpy())
            candidate_errors.append((candidate_prediction.float() - target_model).abs().mean((2, 3, 4)).cpu().numpy())
            previous = torch.cat([context_model[:, -1:], target_model[:, :-1]], dim=1)
            true_motion.append((target_model - previous).abs().mean((1, 2, 3, 4)).cpu().numpy())
            trajectory = torch.cat([history_device[:, -1:], future_device], dim=1)
            velocity = (trajectory[:, 1:] - trajectory[:, :-1]).abs() / std
            displacement = (future_device - history_device[:, -1:]).abs() / std
            features["mean_velocity"].append(velocity.mean((1, 2)).cpu().numpy())
            features["max_displacement"].append(displacement.amax((1, 2)).cpu().numpy())
            features["end_displacement"].append(displacement[:, -1].mean(1).cpu().numpy())
            raw_span = future_device.amax(1) - future_device.amin(1)
            left = raw_span[:, :6].amax(1)
            right = raw_span[:, 7:13].amax(1)
            arms.append((right > left).long().cpu().numpy())
    base_errors = np.concatenate(base_errors)
    candidate_errors = np.concatenate(candidate_errors)
    true_motion = np.concatenate(true_motion)
    arms = np.concatenate(arms)
    features = {key: np.concatenate(value) for key, value in features.items()}
    high_motion = true_motion >= 0.04

    searches = []
    best = None
    tolerance = 1e-9
    for feature_name, values in features.items():
        thresholds = {}
        for arm in (0, 1):
            selected = values[arms == arm]
            thresholds[arm] = np.unique(np.concatenate(([-np.inf], np.quantile(selected, np.linspace(0, 1, 25)), [np.inf])))
        for threshold0 in thresholds[0]:
            for threshold1 in thresholds[1]:
                use_candidate = values <= np.where(arms == 0, threshold0, threshold1)
                for candidate_start_frame in range(9):
                    temporal_gate = np.arange(8) >= candidate_start_frame
                    routed = np.where(use_candidate[:, None] & temporal_gate[None], candidate_errors, base_errors)
                    safe_horizon = bool(np.all(routed.mean(0) <= base_errors.mean(0) + tolerance))
                    horizon_gains = np.asarray([
                        (base_errors[:, index].mean() - routed[:, index].mean())
                        / base_errors[:, index].mean() * 100.0 for index in range(8)
                    ])
                    robust_active_horizon = bool(
                        not temporal_gate.any()
                        or np.all(horizon_gains[temporal_gate] >= args.min_active_horizon_improvement)
                    )
                    safe_arms = all(routed[arms == arm].mean() <= base_errors[arms == arm].mean() + tolerance for arm in (0, 1))
                    safe_high = routed[high_motion].mean() <= base_errors[high_motion].mean() + tolerance
                    if not (safe_horizon and robust_active_horizon and safe_arms and safe_high):
                        continue
                    gain = improvement(base_errors, routed)
                    record = {
                        "feature": feature_name,
                        "threshold_arm0": float(threshold0),
                        "threshold_arm1": float(threshold1),
                        "candidate_start_frame_zero_based": candidate_start_frame,
                        "candidate_active_horizons": list(range(candidate_start_frame + 1, 9)),
                        "candidate_window_count": int(use_candidate.sum()),
                        "candidate_window_fraction": float(use_candidate.mean()),
                        "candidate_frame_fraction": float(use_candidate.mean() * temporal_gate.mean()),
                        "routed_rgb_improvement_percent": gain,
                        "high_motion_improvement_percent": improvement(base_errors[high_motion], routed[high_motion]),
                        "arm0_improvement_percent": improvement(base_errors[arms == 0], routed[arms == 0]),
                        "arm1_improvement_percent": improvement(base_errors[arms == 1], routed[arms == 1]),
                        "minimum_active_horizon_improvement_percent": args.min_active_horizon_improvement,
                        "horizon_improvement_percent": horizon_gains.tolist(),
                    }
                    searches.append(record)
                    if best is None or gain > best[0]:
                        best = (gain, record, use_candidate.copy(), routed.copy())
    if best is None:
        raise RuntimeError("identity route unexpectedly failed safety constraints")
    _, route, use_candidate, routed = best
    report = {
        "format": "track2-v26.3-action-transfer-router",
        "base": str(base_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "windows": len(dataset),
        "base_rgb_mae": float(base_errors.mean()),
        "candidate_rgb_mae": float(candidate_errors.mean()),
        "candidate_rgb_improvement_percent": improvement(base_errors, candidate_errors),
        "candidate_high_motion_improvement_percent": improvement(base_errors[high_motion], candidate_errors[high_motion]),
        "selected_route": route,
        "routed_rgb_mae": float(routed.mean()),
        "safe_route_count": len(searches),
        "action_features": {key: {"min": float(value.min()), "max": float(value.max()), "mean": float(value.mean())} for key, value in features.items()},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "routing_cache.npz", base_errors=base_errors, candidate_errors=candidate_errors,
        routed_errors=routed, true_motion=true_motion, arms=arms, use_candidate=use_candidate,
        **features,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
