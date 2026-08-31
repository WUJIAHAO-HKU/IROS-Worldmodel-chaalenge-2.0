#!/usr/bin/env python3
"""External evaluation and structure-band routing for the v13.8 RGB refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_rgb_refiner_v138 import ContactRGBRefinerV138
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from train_contact_occlusion_head_v131 import active_arm


def frames(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).permute(0, 3, 1, 2).to(device).float() / 255


def alpha_band(source_label: np.ndarray, parent_label: np.ndarray,
               predicted_label: np.ndarray, radius: int) -> np.ndarray:
    source = np.isin(source_label, (2, 3))
    values = []
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    for time in range(8):
        support = source | np.isin(parent_label[time], (2, 3)) | np.isin(predicted_label[time], (2, 3))
        support = cv2.dilate(support.astype(np.uint8), kernel).astype(np.float32)
        support = cv2.GaussianBlur(support, (0, 0), 2.0)
        values.append(np.clip(support, 0, 1))
    return np.stack(values)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    y0, y1, x0, x1 = CONTACT_REGION
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    crop_error = error[:, :, y0:y1, x0:x1]
    truth_label = np.stack([structure_semantic_mask(value[:, y0:y1, x0:x1]) for value in target])
    structure = np.isin(truth_label, (2, 3))[:, :, :, :, None]
    return {"rgb_mae": float(error.mean()), "contact_rgb_mae": float(crop_error.mean()),
            "structure_rgb_mae": float(crop_error[structure.repeat(3, axis=4)].mean()),
            "frame_rgb_mae": error.mean(axis=(0, 2, 3, 4)).tolist(),
            "frame_contact_rgb_mae": crop_error.mean(axis=(0, 2, 3, 4)).tolist()}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True); parser.add_argument("--probability-cache", required=True)
    parser.add_argument("--windows", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--output-cache", required=True)
    parser.add_argument("--band-radius", type=int, default=5); parser.add_argument("--minimum-decay", type=float, default=0.88)
    parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.probability_cache, allow_pickle=False) as cache:
        probability, probability_names = cache["probability"].astype(np.float32), cache["windows"].astype(str)
    if not np.array_equal(names, probability_names): raise ValueError("cache window order differs")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device); model = ContactRGBRefinerV138(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.to(device).eval()
    mean, std = checkpoint["action_mean"].numpy(), checkpoint["action_std"].numpy()
    y0, y1, x0, x1 = CONTACT_REGION; output = parent.copy(); arms = []; diagnostics = []
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        arms.append(arm); action = np.clip((action - mean[arm]) / std[arm], -5, 5)
        parent_crop = parent[index, :, y0:y1, x0:x1]
        last_crop = context[index, -1, y0:y1, x0:x1]
        source_label = structure_semantic_mask(last_crop[None])[0]
        parent_label = structure_semantic_mask(parent_crop)
        predicted_label = probability[index].argmax(axis=1)
        source_semantic = np.stack([source_label == label for label in (1, 2, 3)]).astype(np.float32)
        parent_semantic = np.stack([parent_label == label for label in (1, 2, 3)], axis=1).astype(np.float32)
        source_area = max(int(np.isin(source_label, (2, 3)).sum()), 1)
        ratios = np.isin(parent_label, (2, 3)).sum(axis=(1, 2)) / source_area
        accepted = bool(ratios.min() < args.minimum_decay)
        if accepted:
            refined = model(
                frames(last_crop[None], device),
                torch.from_numpy(parent_crop).permute(0, 3, 1, 2)[None].to(device).float() / 255,
                torch.from_numpy(action)[None].to(device).float(), torch.tensor([arm], device=device),
                torch.from_numpy(source_semantic)[None].to(device),
                torch.from_numpy(parent_semantic)[None].to(device),
            )[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
            alpha = alpha_band(source_label, parent_label, predicted_label, args.band_radius)[..., None]
            blended = parent_crop * (1 - alpha) + refined * alpha
            output[index, :, y0:y1, x0:x1] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        diagnostics.append({"window": name, "arm": arm, "accepted": accepted,
                            "parent_structure_area_ratios": ratios.tolist()})
    arms = np.asarray(arms)
    report = {"format": "track2-contact-rgb-refiner-v13.8-external-eval",
              "checkpoint_step": int(checkpoint["step"]), "sample_count": len(parent),
              "accepted_count": int(sum(item["accepted"] for item in diagnostics)),
              "parameters": {"band_radius": args.band_radius, "minimum_decay": args.minimum_decay},
              "overall": {"parent": metrics(parent, target), "refined": metrics(output, target)},
              "arms": {}, "windows": []}
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm)
        report["arms"][f"arm{arm}"] = {"sample_count": len(indices),
                                          "parent": metrics(parent[indices], target[indices]),
                                          "refined": metrics(output[indices], target[indices])}
    for index, detail in enumerate(diagnostics):
        detail = dict(detail); detail["parent"] = metrics(parent[index:index + 1], target[index:index + 1])
        detail["refined"] = metrics(output[index:index + 1], target[index:index + 1])
        report["windows"].append(detail)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    cache_path = Path(args.output_cache); cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, prediction=output, target=target, context=context,
                        windows=names, arm_id=arms)
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}, indent=2))


if __name__ == "__main__": main()
