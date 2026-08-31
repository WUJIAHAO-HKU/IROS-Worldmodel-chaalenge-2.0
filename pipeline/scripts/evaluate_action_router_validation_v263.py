#!/usr/bin/env python3
"""Evaluate the frozen v26.3 action router on an exact held-out window list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluate_dual_tiny_experts_v150 import MetricEvaluator, arm_metrics
from train_autoregressive_unet import WindowDataset, frames_for_model, rollout
from wam_pipeline.autoregressive_unet import OneStepActionUNet
from wam_pipeline.canonical_arm_texture_v11 import REGIONS
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask


def load_model(path: Path, device: torch.device) -> OneStepActionUNet:
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError(f"unsupported checkpoint: {path}")
    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def improve(base: float, value: float) -> float:
    return (base - value) / base * 100.0


def improvements(base: dict, value: dict) -> dict:
    result = {}
    for key in (
        "rgb_mae", "contact_rgb_mae", "structure_rgb_mae",
        "target_texture_rgb_mae", "active_arm_texture_rgb_mae",
    ):
        result[key.replace("_mae", "_improvement_percent")] = improve(base[key], value[key])
    result["frame_rgb_improvement_percent"] = [
        improve(left, right) for left, right in zip(base["frame_rgb_mae"], value["frame_rgb_mae"])
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--router-report", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--selection-cache", required=True)
    parser.add_argument("--comparison-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--protect-base-structure-radius", type=int, default=0,
        help="Copy parent-predicted black/gray support pixels, dilated by this radius, into routed output.",
    )
    parser.add_argument(
        "--protect-base-contact", action="store_true",
        help="Keep the complete validated contact crop pixel-identical to the parent.",
    )
    parser.add_argument(
        "--protect-base-arm-regions", action="store_true",
        help="Keep both validated arm/logo rectangles pixel-identical to the parent.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_path, candidate_path = Path(args.base), Path(args.candidate)
    with np.load(base_path / "action_normalization.npz", allow_pickle=False) as values:
        mean = torch.from_numpy(np.asarray(values["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(values["std"], dtype=np.float32)).to(device)
    route = json.loads(Path(args.router_report).read_text())["selected_route"]
    if route["feature"] != "mean_velocity":
        raise ValueError("this evaluator currently supports the selected mean_velocity route")

    with np.load(args.selection_cache, allow_pickle=False) as values:
        selected_names = values["windows"].astype(str).tolist()
    episodes = sorted({int(name.split("_")[0][7:]) for name in selected_names})
    dataset = WindowDataset(Path(args.windows), episodes)
    name_to_index = {path.name: index for index, path in enumerate(dataset.paths)}
    missing = [name for name in selected_names if name not in name_to_index]
    if missing:
        raise FileNotFoundError(f"selection contains missing windows: {missing[:5]}")
    selected = Subset(dataset, [name_to_index[name] for name in selected_names])
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    base_model, candidate_model = load_model(base_path, device), load_model(candidate_path, device)

    bases, candidates, targets, contexts, arms, features, motions = [], [], [], [], [], [], []
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for raw_context, history, future, raw_target in loader:
            context = frames_for_model(raw_context).to(device)
            target = frames_for_model(raw_target).to(device)
            history = history.to(device)
            future = future.to(device)
            normalized_history = ((history - mean) / std).float()
            normalized_future = ((future - mean) / std).float()
            base = rollout(base_model, context.clone(), normalized_history, normalized_future)
            candidate = rollout(candidate_model, context.clone(), normalized_history, normalized_future)
            trajectory = torch.cat([history[:, -1:], future], dim=1)
            velocity = (trajectory[:, 1:] - trajectory[:, :-1]).abs() / std
            feature = velocity.mean((1, 2))
            span = future.amax(1) - future.amin(1)
            arm = (span[:, 7:13].amax(1) > span[:, :6].amax(1)).long()
            previous = torch.cat([context[:, -1:], target[:, :-1]], dim=1)
            motion = (target - previous).abs().mean((1, 2, 3, 4))
            bases.append(base.float().cpu())
            candidates.append(candidate.float().cpu())
            targets.append(raw_target)
            contexts.append(raw_context)
            arms.append(arm.cpu())
            features.append(feature.cpu())
            motions.append(motion.cpu())

    base_float = torch.cat(bases).numpy()
    candidate_float = torch.cat(candidates).numpy()
    target = torch.cat(targets).numpy()
    context = torch.cat(contexts).numpy()
    arm = torch.cat(arms).numpy()
    feature = torch.cat(features).numpy()
    true_motion = torch.cat(motions).numpy()
    thresholds = np.where(
        arm == 0, float(route["threshold_arm0"]), float(route["threshold_arm1"])
    )
    use_candidate = feature <= thresholds
    candidate_start = int(route.get("candidate_start_frame_zero_based", 0))
    temporal_gate = np.arange(candidate_float.shape[1]) >= candidate_start
    use_candidate_frames = use_candidate[:, None] & temporal_gate[None]
    routed_float = np.where(
        use_candidate_frames[:, :, None, None, None], candidate_float, base_float
    )
    to_uint8 = lambda value: np.clip(np.round(value.transpose(0, 1, 3, 4, 2) * 255.0), 0, 255).astype(np.uint8)
    base, candidate, routed = map(to_uint8, (base_float, candidate_float, routed_float))
    protected_pixel_fraction = 0.0
    if args.protect_base_structure_radius:
        y0, y1, x0, x1 = CONTACT_REGION
        labels = np.stack([
            structure_semantic_mask(sequence[:, y0:y1, x0:x1]) for sequence in base
        ])
        protected = np.isin(labels, (2, 3))
        radius = args.protect_base_structure_radius
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        protected = np.stack([
            [cv2.dilate(frame.astype(np.uint8), kernel) > 0 for frame in sequence]
            for sequence in protected
        ])
        routed_crop = routed[:, :, y0:y1, x0:x1]
        base_crop = base[:, :, y0:y1, x0:x1]
        routed_crop[protected] = base_crop[protected]
        protected_pixel_fraction = float(protected.mean())
    if args.protect_base_contact:
        y0, y1, x0, x1 = CONTACT_REGION
        routed[:, :, y0:y1, x0:x1] = base[:, :, y0:y1, x0:x1]
        protected_pixel_fraction = 1.0
    if args.protect_base_arm_regions:
        for y0, y1, x0, x1 in REGIONS.values():
            routed[:, :, y0:y1, x0:x1] = base[:, :, y0:y1, x0:x1]

    evaluator = MetricEvaluator(target, arm)
    base_metrics = evaluator(base)
    candidate_metrics = evaluator(candidate)
    routed_metrics = evaluator(routed)
    report = {
        "format": "track2-v26.3-action-router-validation64",
        "frozen_router": route,
        "sample_count": len(selected_names),
        "episodes": episodes,
        "candidate_window_count": int(use_candidate.sum()),
        "candidate_window_fraction": float(use_candidate.mean()),
        "candidate_start_frame_zero_based": candidate_start,
        "candidate_frame_fraction": float(use_candidate_frames.mean()),
        "protect_base_structure_radius": args.protect_base_structure_radius,
        "protect_base_contact": args.protect_base_contact,
        "protect_base_arm_regions": args.protect_base_arm_regions,
        "protected_contact_pixel_fraction": protected_pixel_fraction,
        "base_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "routed_metrics": routed_metrics,
        "candidate_improvements": improvements(base_metrics, candidate_metrics),
        "routed_improvements": improvements(base_metrics, routed_metrics),
        "base_arms": arm_metrics(base, evaluator, arm),
        "routed_arms": arm_metrics(routed, evaluator, arm),
    }
    for side in (0, 1):
        key = f"arm{side}"
        report["routed_arms"][key]["improvements"] = improvements(
            report["base_arms"][key], report["routed_arms"][key]
        )
    high = true_motion >= 0.04
    base_error = np.abs(base.astype(np.float32) - target).mean((1, 2, 3, 4))
    candidate_error = np.abs(candidate.astype(np.float32) - target).mean((1, 2, 3, 4))
    routed_error = np.abs(routed.astype(np.float32) - target).mean((1, 2, 3, 4))
    report["high_motion"] = {
        "sample_count": int(high.sum()),
        "base_rgb_mae": float(base_error[high].mean()),
        "candidate_rgb_mae": float(candidate_error[high].mean()),
        "routed_rgb_mae": float(routed_error[high].mean()),
        "candidate_improvement_percent": improve(base_error[high].mean(), candidate_error[high].mean()),
        "routed_improvement_percent": improve(base_error[high].mean(), routed_error[high].mean()),
    }
    if args.comparison_cache:
        with np.load(args.comparison_cache, allow_pickle=False) as values:
            comparison_names = values["windows"].astype(str).tolist()
            if comparison_names != selected_names:
                raise ValueError("comparison cache window order differs from selection")
            comparison = values["prediction"]
        comparison_metrics = evaluator(comparison)
        report["comparison_cache"] = str(Path(args.comparison_cache).resolve())
        report["comparison_metrics"] = comparison_metrics
        report["routed_vs_comparison_improvements"] = improvements(comparison_metrics, routed_metrics)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation64_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "validation64_routed.npz", prediction=routed, target=target, context=context,
        windows=np.asarray(selected_names), arm_id=arm, action_mean_velocity=feature,
        use_candidate=use_candidate,
    )
    print(json.dumps({
        "candidate_windows": int(use_candidate.sum()),
        "routed_improvements": report["routed_improvements"],
        "high_motion": report["high_motion"],
        "routed_vs_comparison": report.get("routed_vs_comparison_improvements"),
    }, indent=2))


if __name__ == "__main__":
    main()
