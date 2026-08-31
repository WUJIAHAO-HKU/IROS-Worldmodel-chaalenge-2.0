#!/usr/bin/env python3
"""Evaluate the v11 canonical arm-texture renderer on a cached prediction set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import (
    CanonicalArmTextureV11,
    REGIONS,
    RenderParameters,
    _beam_support,
)


def texture_mask(frame: np.ndarray) -> np.ndarray:
    output = np.zeros(frame.shape[:2], dtype=bool)
    for y0, y1, x0, x1 in REGIONS.values():
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
        beam = _beam_support(gray)
        candidate = ((gray > 0.48) & ((gray - blur) > 0.10) & (beam > 0)).astype(np.uint8)
        candidate[:12] = 0
        candidate[112:] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        accepted = np.zeros_like(candidate)
        for index in range(1, count):
            _, _, width, height, area = stats[index]
            if 2 <= area <= 90 and width <= 24 and height <= 18:
                accepted[labels == index] = 1
        accepted = cv2.dilate(accepted, np.ones((5, 5), np.uint8))
        output[y0:y1, x0:x1] |= accepted > 0
    return output


def highpass_batch(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float32)
    for index, frame in enumerate(value):
        source = frame.astype(np.float32) / 255.0
        output[index] = source - cv2.GaussianBlur(source, (0, 0), 1.4)
    return output


def summarize(prediction: np.ndarray, target: np.ndarray, context: np.ndarray) -> dict:
    prediction_f = prediction.astype(np.float32)
    target_f = target.astype(np.float32)
    error = np.abs(prediction_f - target_f)
    previous = np.concatenate((context[-1:], target[:-1]), axis=0).astype(np.float32)
    moving = np.abs(target_f - previous).mean(axis=3) >= 0.03 * 255.0
    dark = target_f.mean(axis=3) < 0.30 * 255.0
    arm = np.zeros(target.shape[:3], dtype=bool)
    arm[:, :150] = target_f[:, :150].mean(axis=3) < 0.55 * 255.0
    masks = np.stack([texture_mask(frame) for frame in target])

    def masked_mae(mask: np.ndarray) -> float:
        return float(error[mask].mean()) if mask.any() else float("nan")

    edge_sum = edge_count = 0.0
    for axis in (1, 2):
        edge_error = np.abs(np.diff(prediction_f, axis=axis) - np.diff(target_f, axis=axis))
        edge_sum += float(edge_error.sum())
        edge_count += edge_error.size
    pred_dark = prediction_f[:, :150].mean(axis=3) < 0.42 * 255.0
    target_dark = target_f[:, :150].mean(axis=3) < 0.42 * 255.0
    structure_iou = float(np.logical_and(pred_dark, target_dark).sum() / max(np.logical_or(pred_dark, target_dark).sum(), 1))
    return {
        "rgb_mae": float(error.mean()),
        "moving_rgb_mae": masked_mae(moving),
        "highpass_mae": float(np.abs(highpass_batch(prediction) - highpass_batch(target)).mean() * 255.0),
        "edge_mae": edge_sum / edge_count,
        "dark_rgb_mae": masked_mae(dark),
        "upper_arm_rgb_mae": masked_mae(arm),
        "target_texture_rgb_mae": masked_mae(masks),
        "dark_structure_iou": structure_iou,
        "target_texture_pixel_fraction": float(masks.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache")
    parser.add_argument("--reference-cache", help="Optional v8/reference cache for end-to-end metrics.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--texture-alpha", type=float, default=1.0)
    parser.add_argument("--minimum-ecc", type=float, default=0.55)
    parser.add_argument("--minimum-overlap", type=float, default=0.20)
    parser.add_argument("--maximum-descriptor-distance", type=float, default=2.5)
    parser.add_argument("--mask-dilation", type=int, default=2)
    parser.add_argument("--mask-blur", type=float, default=0.8)
    parser.add_argument("--highpass-sigma", type=float, default=1.4)
    parser.add_argument("--allow-history-visible", action="store_true")
    parser.add_argument(
        "--texture-mode", choices=("highpass", "positive", "direct", "semantic"), default="highpass"
    )
    parser.add_argument("--inward-shift", type=float, default=0.0)
    parser.add_argument("--vertical-shift", type=float, default=0.0)
    parser.add_argument("--render-sides", choices=("both", "left", "right"), default="both")
    parser.add_argument("--prefer-observed-history", action="store_true")
    parser.add_argument("--observed-history-alpha", type=float)
    args = parser.parse_args()

    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        prediction = cache["prediction"]
        target = cache["target"]
        context = cache["context"]
        names = cache["windows"].astype(str)
    if args.limit:
        prediction, target, context, names = (
            value[: args.limit] for value in (prediction, target, context, names)
        )
    renderer = CanonicalArmTextureV11(args.atlas)
    parameters = RenderParameters(
        top_k=args.top_k,
        texture_alpha=args.texture_alpha,
        minimum_ecc=args.minimum_ecc,
        minimum_overlap=args.minimum_overlap,
        maximum_descriptor_distance=args.maximum_descriptor_distance,
        mask_dilation=args.mask_dilation,
        mask_blur=args.mask_blur,
        highpass_sigma=args.highpass_sigma,
        require_disocclusion=not args.allow_history_visible,
        texture_mode=args.texture_mode,
        inward_shift=args.inward_shift,
        vertical_shift=args.vertical_shift,
        render_sides=args.render_sides,
        prefer_observed_history=args.prefer_observed_history,
        observed_history_alpha=args.observed_history_alpha,
    )
    rendered = np.empty_like(prediction)
    diagnostics = []
    for window_index in range(len(prediction)):
        renderer.reset_observed_state()
        window_diagnostics = []
        for frame_index in range(prediction.shape[1]):
            rendered[window_index, frame_index], frame_diagnostics = renderer.render(
                prediction[window_index, frame_index], parameters, context[window_index]
            )
            window_diagnostics.append(frame_diagnostics)
        diagnostics.append(window_diagnostics)
        if (window_index + 1) % 8 == 0:
            print(f"rendered {window_index + 1}/{len(prediction)} windows", flush=True)

    # Compute temporal masks per window; flattening context cannot provide the
    # preceding target for frames t+2..t+8.
    baseline_parts = [summarize(prediction[i], target[i], context[i]) for i in range(len(prediction))]
    rendered_parts = [summarize(rendered[i], target[i], context[i]) for i in range(len(prediction))]
    baseline = {key: float(np.nanmean([part[key] for part in baseline_parts])) for key in baseline_parts[0]}
    result_metrics = {key: float(np.nanmean([part[key] for part in rendered_parts])) for key in rendered_parts[0]}
    improvements = {}
    for key in baseline:
        if key.endswith("_fraction"):
            continue
        if key.endswith("_iou"):
            improvements[key + "_absolute_gain"] = result_metrics[key] - baseline[key]
        else:
            improvements[key + "_improvement_percent"] = 100.0 * (
                baseline[key] - result_metrics[key]
            ) / max(baseline[key], 1e-12)
    flat_diagnostics = [side for window in diagnostics for frame in window for side in frame]
    accepted = [value for value in flat_diagnostics if value["accepted"]]
    result = {
        "format": "track2-canonical-arm-texture-v11-eval",
        "sample_count": len(prediction),
        "prediction_cache": str(Path(args.prediction_cache).resolve()),
        "atlas": str(Path(args.atlas).resolve()),
        "parameters": vars(parameters),
        "baseline": baseline,
        "rendered": result_metrics,
        "improvements": improvements,
        "accepted_side_fraction": len(accepted) / max(len(flat_diagnostics), 1),
        "mean_edited_fraction_when_accepted": float(
            np.mean([value["edited_fraction"] for value in accepted]) if accepted else 0.0
        ),
        "windows": [
            {"window": str(name), "frames": values} for name, values in zip(names, diagnostics)
        ],
    }
    if args.reference_cache:
        with np.load(args.reference_cache, allow_pickle=False) as reference_cache:
            reference_prediction = reference_cache["prediction"]
            reference_names = reference_cache["windows"].astype(str)
        if not np.array_equal(names, reference_names):
            raise ValueError("reference cache window order differs")
        reference_parts = [
            summarize(reference_prediction[i], target[i], context[i]) for i in range(len(prediction))
        ]
        reference = {
            key: float(np.nanmean([part[key] for part in reference_parts])) for key in reference_parts[0]
        }
        versus_reference = {}
        for key in reference:
            if key.endswith("_fraction"):
                continue
            if key.endswith("_iou"):
                versus_reference[key + "_absolute_gain"] = result_metrics[key] - reference[key]
            else:
                versus_reference[key + "_improvement_percent"] = 100.0 * (
                    reference[key] - result_metrics[key]
                ) / max(reference[key], 1e-12)
        result["reference_cache"] = str(Path(args.reference_cache).resolve())
        result["reference"] = reference
        result["improvements_vs_reference"] = versus_reference
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_cache:
        output_cache = Path(args.output_cache)
        output_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_cache, prediction=rendered, target=target, context=context, windows=names)
    print(json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
