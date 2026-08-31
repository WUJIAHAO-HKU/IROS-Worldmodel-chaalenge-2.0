#!/usr/bin/env python3
"""Evaluate contact-aware rigid layer reconstruction on cached predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_layer_v12 import (
    CONTACT_REGION,
    ContactLayerV12,
    ContactParameters,
    bottle_mask,
    gripper_mask,
)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    y0, y1, x0, x1 = CONTACT_REGION
    pred_crop = prediction[:, y0:y1, x0:x1]
    target_crop = target[:, y0:y1, x0:x1]
    contact_error = np.abs(pred_crop.astype(np.float32) - target_crop.astype(np.float32))
    dark_target = target_crop.mean(axis=3) < 85
    dark_pred = pred_crop.mean(axis=3) < 107
    bottle_target = np.stack([bottle_mask(frame) > 0 for frame in target_crop])
    bottle_pred = np.stack([bottle_mask(frame) > 0 for frame in pred_crop])
    gripper_target = np.stack(
        [gripper_mask(frame, bottle.astype(np.uint8)) > 0 for frame, bottle in zip(target_crop, bottle_target)]
    )
    gripper_pred = np.stack(
        [gripper_mask(frame, bottle.astype(np.uint8)) > 0 for frame, bottle in zip(pred_crop, bottle_pred)]
    )
    dark_intersection = np.logical_and(dark_target, dark_pred).sum()
    gripper_intersection = np.logical_and(gripper_target, gripper_pred).sum()
    gripper_union = np.logical_or(gripper_target, gripper_pred).sum()
    return {
        "rgb_mae": float(error.mean()),
        "contact_rgb_mae": float(contact_error.mean()),
        "contact_dark_recall": float(dark_intersection / max(dark_target.sum(), 1)),
        "contact_dark_precision": float(dark_intersection / max(dark_pred.sum(), 1)),
        "bottle_recall": float(np.logical_and(bottle_target, bottle_pred).sum() / max(bottle_target.sum(), 1)),
        "bottle_iou": float(
            np.logical_and(bottle_target, bottle_pred).sum()
            / max(np.logical_or(bottle_target, bottle_pred).sum(), 1)
        ),
        "gripper_recall": float(gripper_intersection / max(gripper_target.sum(), 1)),
        "gripper_precision": float(gripper_intersection / max(gripper_pred.sum(), 1)),
        "gripper_iou": float(gripper_intersection / max(gripper_union, 1)),
        "gripper_area_ratio": float(gripper_pred.sum() / max(gripper_target.sum(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache")
    parser.add_argument("--bottle-alpha", type=float, default=0.72)
    parser.add_argument("--gripper-alpha", type=float, default=0.88)
    parser.add_argument("--minimum-ecc", type=float, default=0.50)
    parser.add_argument("--minimum-overlap", type=float, default=0.42)
    parser.add_argument("--minimum-gripper-area", type=int, default=45)
    parser.add_argument("--minimum-query-gripper-area", type=int, default=45)
    parser.add_argument("--maximum-gripper-area", type=int, default=100000)
    parser.add_argument("--edge-blur", type=float, default=0.55)
    parser.add_argument("--gripper-alignment", choices=("centroid", "top_right"), default="top_right")
    parser.add_argument("--maximum-gripper-area-ratio", type=float, default=0.78)
    parser.add_argument(
        "--reconstruction-mode",
        choices=("query_sharpen", "hard_trimap", "rigid_reprojection"),
        default="query_sharpen",
    )
    parser.add_argument("--structure-dilation", type=int, default=0)
    parser.add_argument("--trimap-radius", type=int, default=1)
    parser.add_argument("--green-purity-ratio", type=float, default=0.62)
    args = parser.parse_args()
    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        prediction = cache["prediction"]
        target = cache["target"]
        context = cache["context"]
        names = cache["windows"].astype(str)
    renderer = ContactLayerV12()
    parameters = ContactParameters(
        bottle_alpha=args.bottle_alpha,
        gripper_alpha=args.gripper_alpha,
        minimum_ecc=args.minimum_ecc,
        minimum_bottle_overlap=args.minimum_overlap,
        minimum_gripper_area=args.minimum_gripper_area,
        minimum_query_gripper_area=args.minimum_query_gripper_area,
        maximum_gripper_area=args.maximum_gripper_area,
        edge_blur=args.edge_blur,
        gripper_alignment=args.gripper_alignment,
        maximum_gripper_area_ratio=args.maximum_gripper_area_ratio,
        reconstruction_mode=args.reconstruction_mode,
        structure_dilation=args.structure_dilation,
        trimap_radius=args.trimap_radius,
        green_purity_ratio=args.green_purity_ratio,
    )
    rendered = np.empty_like(prediction)
    diagnostics = []
    for window_index in range(len(prediction)):
        frame_diagnostics = []
        for frame_index in range(prediction.shape[1]):
            rendered[window_index, frame_index], detail = renderer.render(
                prediction[window_index, frame_index], context[window_index], parameters
            )
            frame_diagnostics.append(detail)
        diagnostics.append(frame_diagnostics)
    baseline_parts = [metrics(prediction[i], target[i]) for i in range(len(prediction))]
    rendered_parts = [metrics(rendered[i], target[i]) for i in range(len(prediction))]
    baseline = {key: float(np.mean([value[key] for value in baseline_parts])) for key in baseline_parts[0]}
    result_metrics = {key: float(np.mean([value[key] for value in rendered_parts])) for key in rendered_parts[0]}
    result = {
        "format": "track2-contact-layer-v12-eval",
        "sample_count": len(prediction),
        "parameters": vars(parameters),
        "baseline": baseline,
        "rendered": result_metrics,
        "delta": {key: result_metrics[key] - baseline[key] for key in baseline},
        "accepted_frame_fraction": float(np.mean([[frame["accepted"] for frame in window] for window in diagnostics])),
        "windows": [
            {"window": str(name), "frames": detail} for name, detail in zip(names, diagnostics)
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_cache:
        np.savez_compressed(
            args.output_cache, prediction=rendered, target=target, context=context, windows=names
        )
    print(json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
