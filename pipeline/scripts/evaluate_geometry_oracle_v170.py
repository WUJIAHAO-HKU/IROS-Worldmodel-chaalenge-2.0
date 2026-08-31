#!/usr/bin/env python3
"""Measure the v17 pose/layer/Z-buffer transport ceiling on a disjoint dev set.

Future RGB-derived visibility masks and pixel routing are used only to measure
the attainable ceiling.  The transported RGB itself always comes from one of
the five observed frames.  Both exact recorded poses and the learned action
projector are evaluated with the same object-layer compositor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import (
    REGIONS,
    _observed_beam_mask,
    _observed_logo_mask,
)
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_masks, structure_semantic_mask
from wam_pipeline.object_geometry_v170 import (
    ActionPoseProjector,
    affine_from_landmarks,
    normalize_action,
    oracle_update,
    project_endpose,
    warp_layer,
)


def parse_name(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"episode(\d+)_(\d+)\.npz", name)
    if match is None:
        raise ValueError(name)
    return int(match.group(1)), int(match.group(2))


def active_arm(history: np.ndarray, future: np.ndarray) -> int:
    sequence = np.concatenate((history[-1:], future), axis=0)
    motion = np.abs(np.diff(sequence, axis=0)).mean(0)
    return int(motion[7:].mean() > motion[:7].mean())


def full_mask(frame: np.ndarray, side: str, kind: str) -> np.ndarray:
    output = np.zeros(frame.shape[:2], dtype=np.uint8)
    if kind == "beam":
        y0, y1, x0, x1 = REGIONS[side]
        local = _observed_beam_mask(frame[y0:y1, x0:x1])
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        local = cv2.dilate(local, np.ones((3, 3), np.uint8))
        output[y0:y1, x0:x1] = local
        return output
    bottle, black, grey = structure_masks(frame)
    value = {"bottle": bottle, "black": black, "grey": grey}[kind]
    return value.astype(np.uint8)


def predicted_landmarks(checkpoint: dict, actions: np.ndarray, arm: int,
                        device: torch.device) -> np.ndarray:
    model = checkpoint["runtime_model"]
    arm_ids = torch.full((len(actions),), arm, dtype=torch.long, device=device)
    value = torch.from_numpy(actions.astype(np.float32)).to(device)
    action_mean = checkpoint["runtime_action_mean"]
    action_std = checkpoint["runtime_action_std"]
    target_mean = checkpoint["runtime_target_mean"]
    target_std = checkpoint["runtime_target_std"]
    with torch.inference_mode():
        normalized = normalize_action(value, action_mean, action_std, arm_ids)
        result = model(normalized, arm_ids) * target_std[arm_ids] + target_mean[arm_ids]
    return result.cpu().numpy()


def exact_landmarks(handle: h5py.File, frames: list[int], arm: int) -> np.ndarray:
    side = "left" if arm == 0 else "right"
    pose = handle[f"endpose/{side}_endpose"]
    intrinsic = handle["observation/head_camera/intrinsic_cv"]
    extrinsic = handle["observation/head_camera/extrinsic_cv"]
    return np.stack([project_endpose(pose[index], intrinsic[index], extrinsic[index])
                     for index in frames])


def render(
    parent: np.ndarray,
    target: np.ndarray,
    context: np.ndarray,
    source_geometry: np.ndarray,
    target_geometry: np.ndarray,
    arm: int,
) -> tuple[np.ndarray, dict]:
    side = "left" if arm == 0 else "right"
    output = parent.copy(); counts = {kind: 0 for kind in ("beam", "black", "grey", "bottle")}
    for time in range(len(parent)):
        # Future target masks act as perfect z-buffer/visibility supervision.
        target_masks = {kind: full_mask(target[time], side, kind)
                        for kind in ("beam", "black", "grey", "bottle")}
        for source_index in range(len(context)):
            matrix = affine_from_landmarks(source_geometry[source_index], target_geometry[time])
            warped_rgb = warp_layer(context[source_index], matrix, cv2.INTER_CUBIC)
            for kind in ("beam", "black", "grey"):
                source_mask = full_mask(context[source_index], side, kind)
                warped_mask = warp_layer(source_mask, matrix, cv2.INTER_NEAREST) > 0
                valid = warped_mask & (target_masks[kind] > 0)
                output[time], selected = oracle_update(
                    output[time], target[time], warped_rgb, valid, margin=0.25
                )
                counts[kind] += int(selected.sum())

            # Before contact the bottle is static; after contact it follows the
            # active gripper.  Oracle routing compares both hypotheses while RGB
            # is still restricted to observed bottle pixels.
            bottle_mask = full_mask(context[source_index], side, "bottle")
            for bottle_matrix in (np.asarray([[1, 0, 0], [0, 1, 0]], np.float32), matrix):
                bottle_rgb = warp_layer(context[source_index], bottle_matrix, cv2.INTER_CUBIC)
                warped_mask = warp_layer(bottle_mask, bottle_matrix, cv2.INTER_NEAREST) > 0
                valid = warped_mask & (target_masks["bottle"] > 0)
                output[time], selected = oracle_update(
                    output[time], target[time], bottle_rgb, valid, margin=0.25
                )
                counts["bottle"] += int(selected.sum())
    return output, counts


def texture_mask(sequence: np.ndarray, arms: np.ndarray) -> np.ndarray:
    output = np.zeros(sequence.shape[:4], bool)
    for sample in range(len(sequence)):
        side = "left" if arms[sample] == 0 else "right"
        y0, y1, x0, x1 = REGIONS[side]
        for time, frame in enumerate(sequence[sample]):
            local = _observed_logo_mask(frame[y0:y1, x0:x1])
            if local.any():
                output[sample, time, y0:y1, x0:x1] = cv2.dilate(
                    local, np.ones((5, 5), np.uint8)
                ) > 0
    return output


def metrics(prediction: np.ndarray, target: np.ndarray, arms: np.ndarray) -> dict:
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    y0, y1, x0, x1 = CONTACT_REGION
    crop = error[:, :, y0:y1, x0:x1]
    labels = np.stack([structure_semantic_mask(value[:, y0:y1, x0:x1]) for value in target])
    structure = np.repeat(np.isin(labels, (2, 3))[..., None], 3, axis=4)
    text = np.repeat(texture_mask(target, arms)[..., None], 3, axis=4)
    return {
        "rgb_mae": float(error.mean()),
        "contact_rgb_mae": float(crop.mean()),
        "structure_rgb_mae": float(crop[structure].mean()) if structure.any() else None,
        "active_text_rgb_mae": float(error[text].mean()) if text.any() else None,
        "frame_rgb_mae": error.mean(axis=(0, 2, 3, 4)).tolist(),
    }


def improvements(parent: dict, value: dict) -> dict:
    return {key: 100 * (parent[key] - value[key]) / max(parent[key], 1e-12)
            for key in ("rgb_mae", "contact_rgb_mae", "structure_rgb_mae", "active_text_rgb_mae")
            if parent.get(key) is not None and value.get(key) is not None}


def save_sheet(path: Path, context: np.ndarray, parent: np.ndarray, exact: np.ndarray,
               learned: np.ndarray, target: np.ndarray) -> None:
    labels = ("observed t0", "parent", "exact-pose oracle", "learned-pose oracle", "target")
    values = (np.repeat(context[-1:], 8, axis=0), parent, exact, learned, target)
    canvas = Image.new("RGB", (256 * len(labels), 8 * 278), "white")
    draw = ImageDraw.Draw(canvas)
    for time in range(8):
        for column, (label, sequence) in enumerate(zip(labels, values)):
            x, y = column * 256, time * 278
            draw.text((x + 5, y + 4), f"t+{time + 1} {label}", fill="black")
            canvas.paste(Image.fromarray(sequence[time]), (x, y + 22))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="36,47")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--visualizations", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    selected_episodes = {int(value) for value in args.episodes.split(",") if value}
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        names = cache["windows"].astype(str); parent_all = cache["prediction"]
        target_all = cache["target"] if "target" in cache.files else None
        context_all = cache["context"] if "context" in cache.files else None
    indices = [index for index, name in enumerate(names) if parse_name(name)[0] in selected_episodes]
    if args.max_samples > 0 and len(indices) > args.max_samples:
        indices = [indices[value] for value in np.linspace(0, len(indices) - 1,
                                                           args.max_samples, dtype=int)]
    device = torch.device(args.device)
    checkpoint = torch.load(args.pose_checkpoint, map_location=device, weights_only=False)
    model = ActionPoseProjector(checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    checkpoint.update({
        "runtime_model": model,
        "runtime_action_mean": torch.from_numpy(checkpoint["action_mean"]).to(device),
        "runtime_action_std": torch.from_numpy(checkpoint["action_std"]).to(device),
        "runtime_target_mean": torch.from_numpy(checkpoint["target_mean"]).to(device),
        "runtime_target_std": torch.from_numpy(checkpoint["target_std"]).to(device),
    })

    parents = []; targets = []; contexts = []; arms = []; exact_values = []; learned_values = []
    records = []; output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for order, index in enumerate(indices):
        name = names[index]; episode, start = parse_name(name)
        if target_all is None or context_all is None:
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                context = window["context_frames"]; target = window["target_frames"]
                history = window["history_actions"]; future = window["future_actions"]
        else:
            context = context_all[index]; target = target_all[index]
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                history = window["history_actions"]; future = window["future_actions"]
        arm = active_arm(history, future); side_actions = slice(arm * 7, (arm + 1) * 7)
        source_actions = np.concatenate((history[:1], history), axis=0)[:5, side_actions]
        target_actions = future[:, side_actions]
        with h5py.File(Path(args.hdf5) / f"episode{episode}.hdf5", "r") as handle:
            exact_source = exact_landmarks(handle, list(range(start, start + 5)), arm)
            exact_target = exact_landmarks(handle, list(range(start + 5, start + 13)), arm)
        learned_source = predicted_landmarks(checkpoint, source_actions, arm, device)
        learned_target = predicted_landmarks(checkpoint, target_actions, arm, device)
        exact, exact_counts = render(parent_all[index], target, context, exact_source, exact_target, arm)
        learned, learned_counts = render(parent_all[index], target, context,
                                         learned_source, learned_target, arm)
        parents.append(parent_all[index]); targets.append(target); contexts.append(context); arms.append(arm)
        exact_values.append(exact); learned_values.append(learned)
        records.append({"window": name, "arm": arm, "exact_selected_pixels": exact_counts,
                        "learned_selected_pixels": learned_counts})
        if order < args.visualizations:
            save_sheet(output / f"{Path(name).stem}_sheet.png", context, parent_all[index],
                       exact, learned, target)
        print(json.dumps({"completed": order + 1, "total": len(indices), "window": name}), flush=True)

    parents = np.stack(parents); targets = np.stack(targets); contexts = np.stack(contexts)
    arms = np.asarray(arms); exact_values = np.stack(exact_values); learned_values = np.stack(learned_values)
    parent_metrics = metrics(parents, targets, arms)
    exact_metrics = metrics(exact_values, targets, arms)
    learned_metrics = metrics(learned_values, targets, arms)
    result = {
        "format": "track2-object-geometry-zbuffer-oracle-v17.0",
        "warning": "Diagnostic ceiling only: future target masks and target-aware pixel routing are used.",
        "episodes": sorted(selected_episodes),
        "sample_count": len(indices),
        "parent": parent_metrics,
        "exact_pose_oracle": exact_metrics,
        "learned_pose_oracle": learned_metrics,
        "exact_pose_improvement_percent": improvements(parent_metrics, exact_metrics),
        "learned_pose_improvement_percent": improvements(parent_metrics, learned_metrics),
        "records": records,
    }
    temporary = output / f"report.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(result, indent=2) + "\n"); os.replace(temporary, output / "report.json")
    np.savez_compressed(output / "predictions.npz", parent=parents, exact=exact_values,
                        learned=learned_values, target=targets, context=contexts,
                        windows=names[indices], arm_id=arms)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
