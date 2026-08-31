#!/usr/bin/env python3
"""Full dev evaluation and conservative threshold calibration for v17.2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import REGIONS, _observed_logo_mask
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.geometry_visibility_router_v172 import GeometryVisibilityRouterV172, hard_render
from train_geometry_visibility_router_v172 import (
    episode, load_pose, make_example, precompute_geometry, tensors,
)


FIELDS = ("rgb", "contact", "structure", "text")


def empty() -> dict:
    return {**{f"{key}_sum": 0.0 for key in FIELDS},
            **{f"{key}_count": 0 for key in FIELDS}, "selected": 0, "focus": 0}


def masks(target: np.ndarray, arm: int) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = CONTACT_REGION
    label = structure_semantic_mask(target[None, y0:y1, x0:x1])[0]
    structure = np.isin(label, (2, 3))
    text = np.zeros(target.shape[:2], bool); side = "left" if arm == 0 else "right"
    ay0, ay1, ax0, ax1 = REGIONS[side]
    local = _observed_logo_mask(target[ay0:ay1, ax0:ax1])
    if local.any(): text[ay0:ay1, ax0:ax1] = cv2.dilate(local, np.ones((5, 5), np.uint8)) > 0
    return structure, text


def add(state: dict, prediction: np.ndarray, target: np.ndarray,
        structure: np.ndarray, text: np.ndarray, selected: int = 0, focus: int = 0) -> None:
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    y0, y1, x0, x1 = CONTACT_REGION; contact = error[y0:y1, x0:x1]
    values = {"rgb": (error, np.ones(error.shape[:-1], bool)),
              "contact": (contact, np.ones(contact.shape[:-1], bool)),
              "structure": (contact, structure), "text": (error, text)}
    for key, (value, mask) in values.items():
        if mask.any():
            state[f"{key}_sum"] += float(value[mask].sum())
            state[f"{key}_count"] += int(mask.sum() * 3)
    state["selected"] += selected; state["focus"] += focus


def finish(state: dict) -> dict:
    result = {f"{key}_mae": state[f"{key}_sum"] / max(state[f"{key}_count"], 1)
              for key in FIELDS}
    result["selected_fraction"] = state["selected"] / max(state["focus"], 1)
    return result


def improvement(parent: dict, routed: dict) -> dict:
    return {key: 100 * (parent[key] - routed[key]) / max(parent[key], 1e-9)
            for key in parent if key.endswith("_mae")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--pose-checkpoint", required=True); parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--episodes", default="36,47")
    parser.add_argument("--thresholds", default="0,.25,.5,.75,1,1.5,2,3,4,6")
    parser.add_argument("--device", default="cuda"); args = parser.parse_args(); cv2.setNumThreads(1)
    device = torch.device(args.device); windows = Path(args.windows)
    selected_episodes = {int(v) for v in args.episodes.split(",") if v}
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        names_all = cache["windows"].astype(str)
        indices = np.asarray([i for i, name in enumerate(names_all) if episode(name) in selected_episodes])
        names = names_all[indices]; parent = cache["prediction"][indices]
        target = cache["target"][indices]; context = cache["context"][indices]
    pose = load_pose(args.pose_checkpoint, device)
    source_geometry, future_geometry, arms = precompute_geometry(windows, names, pose, device)
    checkpoint = torch.load(args.router_checkpoint, map_location="cpu", weights_only=False)
    model = GeometryVisibilityRouterV172(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    thresholds = [float(v) for v in args.thresholds.split(",")]
    parent_state = empty(); parent_arm_state = {0: empty(), 1: empty()}
    parent_horizon_state = [empty() for _ in range(8)]
    threshold_state = {v: empty() for v in thresholds}
    arm_state = {v: {0: empty(), 1: empty()} for v in thresholds}
    horizon_state = {v: [empty() for _ in range(8)] for v in thresholds}
    predictions = {v: np.empty_like(parent) for v in thresholds}
    with torch.inference_mode():
        for index, name in enumerate(names):
            for time in range(8):
                example = make_example(index, time, parent, target, context,
                                       source_geometry, future_geometry, arms)
                p, t, c, s, condition, horizon, arm = tensors([example], device)
                score = model(p, c, s, condition, horizon, arm)
                structure, text = masks(target[index, time], int(arms[index]))
                add(parent_state, parent[index, time], target[index, time], structure, text)
                add(parent_arm_state[int(arms[index])], parent[index, time], target[index, time], structure, text)
                add(parent_horizon_state[time], parent[index, time], target[index, time], structure, text)
                focus = int((s.max(2).values > .5).any(1).sum())
                for threshold in thresholds:
                    output, choice = hard_render(p, c, s, score, threshold)
                    value = np.round(output[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    predictions[threshold][index, time] = value
                    selected = int((choice > 0).sum())
                    add(threshold_state[threshold], value, target[index, time], structure, text, selected, focus)
                    add(arm_state[threshold][int(arms[index])], value, target[index, time], structure, text, selected, focus)
                    add(horizon_state[threshold][time], value, target[index, time], structure, text, selected, focus)
            if (index + 1) % 8 == 0:
                print(json.dumps({"completed": index + 1, "total": len(names)}), flush=True)
    parent_metrics = finish(parent_state)
    parent_arms = {f"arm{arm}": finish(value) for arm, value in parent_arm_state.items()}
    parent_horizons = [finish(value) for value in parent_horizon_state]; trials = []
    for threshold in thresholds:
        routed = finish(threshold_state[threshold])
        arms_result = {f"arm{arm}": finish(value) for arm, value in arm_state[threshold].items()}
        horizons_result = [finish(value) for value in horizon_state[threshold]]
        gains = improvement(parent_metrics, routed)
        # A safe candidate may not regress any primary metric, arm RGB, or horizon RGB.
        safe = (min(gains.values()) >= 0 and
                all(value["rgb_mae"] <= parent_arms[key]["rgb_mae"]
                    for key, value in arms_result.items()) and
                all(value["rgb_mae"] <= parent_horizons[index]["rgb_mae"]
                    for index, value in enumerate(horizons_result)))
        trials.append({"threshold": threshold, "safe": safe, "metrics": routed,
                       "improvement_percent": gains, "arms": arms_result,
                       "horizons": horizons_result})
    safe = [value for value in trials if value["safe"]]
    choice = max(safe, key=lambda value: min(value["improvement_percent"].values())) if safe else min(
        trials, key=lambda value: value["metrics"]["rgb_mae"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    report = {"format": "track2-geometry-visibility-router-v17.2-calibration",
              "data_boundary": "supplied_50_episodes_only", "episodes": sorted(selected_episodes),
              "sample_count": len(names), "checkpoint_step": checkpoint["step"],
              "parent": parent_metrics, "parent_arms": parent_arms,
              "parent_horizons": parent_horizons, "selected": choice, "trials": trials}
    temporary = output / f"report.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(report, indent=2) + "\n"); os.replace(temporary, output / "report.json")
    np.savez_compressed(output / "predictions.npz", prediction=predictions[choice["threshold"]],
                        parent=parent, target=target, context=context, windows=names, arm_id=arms)
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))


if __name__ == "__main__": main()
