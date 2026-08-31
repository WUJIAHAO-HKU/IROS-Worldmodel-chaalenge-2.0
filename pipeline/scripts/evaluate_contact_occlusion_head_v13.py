#!/usr/bin/env python3
"""Evaluate and render v13 recurrent contact layers on cached predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_layer_v12 import bottle_mask, gripper_mask
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION, ContactOcclusionHeadV13


def semantic_mask(frames: np.ndarray) -> np.ndarray:
    result = []
    for frame in frames:
        bottle = bottle_mask(frame) > 0
        gripper = gripper_mask(frame, bottle.astype(np.uint8)) > 0
        label = bottle.astype(np.uint8); label[gripper] = 2; result.append(label)
    return np.stack(result)


def rgb_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    y0, y1, x0, x1 = CONTACT_REGION
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    contact = error[:, :, y0:y1, x0:x1]
    pred_labels = np.stack([semantic_mask(value[:, y0:y1, x0:x1]) for value in prediction])
    target_labels = np.stack([semantic_mask(value[:, y0:y1, x0:x1]) for value in target])
    output = {"rgb_mae": float(error.mean()), "contact_rgb_mae": float(contact.mean())}
    for name, label in (("bottle", 1), ("gripper", 2)):
        pred, truth = pred_labels == label, target_labels == label
        intersection = np.logical_and(pred, truth).sum()
        output[f"{name}_iou"] = float(intersection / max(np.logical_or(pred, truth).sum(), 1))
        output[f"{name}_recall"] = float(intersection / max(truth.sum(), 1))
        output[f"{name}_precision"] = float(intersection / max(pred.sum(), 1))
    return output


def _pca_transform(source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray | None:
    source = np.argwhere(source_mask)[:, ::-1].astype(np.float64)
    target = np.argwhere(target_mask)[:, ::-1].astype(np.float64)
    if len(source) < 20 or len(target) < 20:
        return None
    source_center, target_center = source.mean(0), target.mean(0)
    source_cov = np.cov((source - source_center).T) + np.eye(2) * 1e-3
    target_cov = np.cov((target - target_center).T) + np.eye(2) * 1e-3
    se, sv = np.linalg.eigh(source_cov); te, tv = np.linalg.eigh(target_cov)
    order_s, order_t = np.argsort(se)[::-1], np.argsort(te)[::-1]
    se, sv = se[order_s], sv[:, order_s]; te, tv = te[order_t], tv[:, order_t]
    # Resolve eigenvector sign ambiguity using a consistent image-axis orientation.
    for axis in range(2):
        if np.dot(sv[:, axis], tv[:, axis]) < 0:
            tv[:, axis] *= -1
    scale = np.sqrt(np.clip(te / se, 0.45, 2.2))
    linear = tv @ np.diag(scale) @ sv.T
    translation = target_center - linear @ source_center
    return np.concatenate((linear, translation[:, None]), axis=1).astype(np.float32)


def _observed_dark_material(source: np.ndarray, source_gripper: np.ndarray,
                            ceiling: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative black-material support and its fallback colour.

    ``gripper_mask`` is deliberately dilated for semantic scoring.  Using that
    dilated mask as RGB texture support also picks up the bright/green pixels
    immediately outside the pad, which become a persistent halo after affine
    reprojection.  Keep only observed neutral-dark pixels and hard-limit their
    value so a semantic foreground pixel can never be painted white.
    """
    value = source.astype(np.float32)
    chroma = value.max(axis=2) - value.min(axis=2)
    support = source_gripper & (value.mean(axis=2) <= ceiling) & (chroma <= 42.0)
    if support.sum() >= 12:
        fallback = np.percentile(value[support], 35, axis=0)
    elif source_gripper.any():
        fallback = np.percentile(value[source_gripper], 20, axis=0)
    else:
        fallback = np.asarray((38, 38, 38), np.float32)
    return support, np.minimum(fallback.astype(np.float32), ceiling)


def _remove_stale_gripper(query: np.ndarray, current_gripper: np.ndarray,
                          current_bottle: np.ndarray, class_probability: np.ndarray,
                          confidence_margin: float, guard_radius: int,
                          inpaint_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint parent-only contact-pad support that the recurrent head rejects."""
    query_u8 = np.clip(np.round(query), 0, 255).astype(np.uint8)
    query_bottle = bottle_mask(query_u8)
    old_gripper = gripper_mask(query_u8, query_bottle) > 0
    diameter = 2 * guard_radius + 1
    guarded = cv2.dilate(current_gripper.astype(np.uint8),
                         np.ones((diameter, diameter), np.uint8)) > 0
    gripper_probability = class_probability[2]
    non_gripper_probability = np.maximum(class_probability[0], class_probability[1])
    stale = old_gripper & ~guarded
    stale &= non_gripper_probability >= gripper_probability + confidence_margin
    # A learned bottle label at the old pad location is especially reliable:
    # it says that the arm moved away and revealed the bottle underneath.
    stale |= old_gripper & ~guarded & current_bottle & (
        class_probability[1] > gripper_probability
    )
    if not stale.any():
        return query, stale
    cleaned = cv2.inpaint(query_u8, stale.astype(np.uint8) * 255,
                          inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned.astype(np.float32), stale


def render_sequence(parent: np.ndarray, context: np.ndarray, probability: np.ndarray,
                    gripper_threshold: float, bottle_threshold: float,
                    green_ratio: float = 0.58, decay_ratio: float = 0.65,
                    recovery_ratio: float = 0.65, material_ceiling: float = 92.0,
                    stale_confidence_margin: float = 0.08,
                    stale_guard_radius: int = 1,
                    stale_inpaint_radius: float = 3.0,
                    paint_erosion: int = 1,
                    paint_kernel: str = "square") -> tuple[np.ndarray, dict]:
    y0, y1, x0, x1 = CONTACT_REGION
    output = parent.copy(); source = context[-1, y0:y1, x0:x1]
    source_bottle = bottle_mask(source) > 0
    source_gripper = gripper_mask(source, source_bottle.astype(np.uint8)) > 0
    if source_bottle.sum() < 220 or source_gripper.sum() < 45:
        return output, {"accepted": False, "reason": "no_observed_contact", "frames": []}
    parent_area_ratios = []
    for frame in parent[:, y0:y1, x0:x1]:
        query_bottle = bottle_mask(frame)
        parent_area_ratios.append(float(gripper_mask(frame, query_bottle).sum() /
                                        max(source_gripper.sum(), 1)))
    # Sequence-level latch: once any future frame loses the observed structure,
    # use the recurrent trajectory for all eight frames.  This preserves temporal
    # continuity without overwriting sequences whose parent is already intact.
    if min(parent_area_ratios) >= decay_ratio:
        return output, {"accepted": False, "reason": "parent_structure_intact",
                        "parent_area_ratios": parent_area_ratios, "frames": []}
    if parent_area_ratios[-1] < recovery_ratio:
        return output, {"accepted": False, "reason": "non_transient_geometry_change",
                        "parent_area_ratios": parent_area_ratios, "frames": []}
    material_support, fallback = _observed_dark_material(
        source, source_gripper, material_ceiling
    )
    frame_details = []
    for time in range(len(parent)):
        query = output[time, y0:y1, x0:x1].astype(np.float32)
        class_probability = probability[time]
        bottle_probability, gripper_probability = class_probability[1], class_probability[2]
        gripper = (gripper_probability >= gripper_threshold) & (gripper_probability > bottle_probability)
        bottle = (bottle_probability >= bottle_threshold) & ~gripper
        # Training labels already contain the one-pixel semantic dilation from
        # gripper_mask().  Paint its eroded core so scoring does not dilate the
        # rendered pad a second time and create an outline around every frame.
        if paint_erosion:
            diameter = 2 * paint_erosion + 1
            if paint_kernel == "cross":
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (diameter, diameter))
            elif paint_kernel == "square":
                kernel = np.ones((diameter, diameter), np.uint8)
            else:
                raise ValueError(f"unknown paint kernel: {paint_kernel}")
            paint_gripper = cv2.erode(
                gripper.astype(np.uint8), kernel
            ) > 0
        else:
            paint_gripper = gripper
        if paint_gripper.sum() < 20:
            paint_gripper = gripper
        query, stale = _remove_stale_gripper(
            query, gripper, bottle, class_probability, stale_confidence_margin,
            stale_guard_radius, stale_inpaint_radius
        )
        transform = _pca_transform(material_support, paint_gripper)
        if transform is not None:
            texture = cv2.warpAffine(source.astype(np.float32), transform, (source.shape[1], source.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            support = cv2.warpAffine(material_support.astype(np.uint8), transform,
                                     (source.shape[1], source.shape[0]), flags=cv2.INTER_NEAREST) > 0
        else:
            texture = np.broadcast_to(fallback, query.shape); support = np.zeros(paint_gripper.shape, bool)
        texture = np.minimum(texture, material_ceiling)
        # Geometry comes from the learned mask; observed material supplies texture where possible.
        query[paint_gripper] = np.where(
            support[paint_gripper, None], texture[paint_gripper], fallback
        )
        ring = (cv2.dilate(paint_gripper.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & bottle
        green = query[..., 1]
        mixed = ring & (green > np.maximum(query[..., 0], query[..., 2]) + 3)
        query[..., 0][mixed] = np.minimum(query[..., 0][mixed], green[mixed] * green_ratio)
        query[..., 2][mixed] = np.minimum(query[..., 2][mixed], green[mixed] * green_ratio)
        output[time, y0:y1, x0:x1] = np.clip(np.round(query), 0, 255).astype(np.uint8)
        frame_details.append({"frame": time, "gripper_pixels": int(gripper.sum()),
                              "paint_gripper_pixels": int(paint_gripper.sum()),
                              "stale_gripper_pixels": int(stale.sum()),
                              "bottle_pixels": int(bottle.sum()), "green_boundary_pixels": int(mixed.sum()),
                              "gripper_mean_probability": float(gripper_probability[gripper].mean()) if gripper.any() else 0.0})
    return output, {"accepted": True, "parent_area_ratios": parent_area_ratios, "frames": frame_details}


@torch.inference_mode()
def infer(model, checkpoint, parent, context, names, windows, device):
    y0, y1, x0, x1 = CONTACT_REGION; probabilities = []
    action_mean = checkpoint["action_mean"].numpy(); action_std = checkpoint["action_std"].numpy()
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as window:
            actions = (window["future_actions"].copy() - action_mean) / action_std
        last = torch.from_numpy(context[index, -1, y0:y1, x0:x1]).permute(2, 0, 1)[None].to(device).float() / 255
        prediction = torch.from_numpy(parent[index, :, y0:y1, x0:x1]).permute(0, 3, 1, 2)[None].to(device).float() / 255
        action = torch.from_numpy(actions)[None].to(device).float()
        probabilities.append(model(last, prediction, action).softmax(2)[0].cpu().numpy())
    return np.stack(probabilities)


def learned_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    y0, y1, x0, x1 = CONTACT_REGION
    truth = np.stack([semantic_mask(value[:, y0:y1, x0:x1]) for value in target])
    predicted = probability.argmax(axis=2)
    # A calibrated threshold is used by the renderer, so report its support too.
    gp = probability[:, :, 2]; bp = probability[:, :, 1]
    rendered_gripper = (gp >= threshold) & (gp > bp)
    result = {}
    for name, label, mask in (("bottle", 1, predicted == 1), ("gripper", 2, rendered_gripper)):
        target_mask = truth == label; intersection = np.logical_and(mask, target_mask).sum()
        result[f"{name}_iou"] = float(intersection / max(np.logical_or(mask, target_mask).sum(), 1))
        result[f"{name}_recall"] = float(intersection / max(target_mask.sum(), 1))
        result[f"{name}_precision"] = float(intersection / max(mask.sum(), 1))
    frame_iou = []
    for time in range(8):
        pred, target_mask = rendered_gripper[:, time], truth[:, time] == 2
        frame_iou.append(float(np.logical_and(pred, target_mask).sum() /
                               max(np.logical_or(pred, target_mask).sum(), 1)))
    result["gripper_frame_iou"] = frame_iou; result["minimum_frame_iou"] = min(frame_iou)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-cache", required=True); parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache"); parser.add_argument("--gripper-threshold", type=float, default=0.55)
    parser.add_argument("--bottle-threshold", type=float, default=0.50); parser.add_argument("--green-ratio", type=float, default=0.58)
    parser.add_argument("--decay-ratio", type=float, default=0.65)
    parser.add_argument("--recovery-ratio", type=float, default=0.65)
    parser.add_argument("--material-ceiling", type=float, default=92.0)
    parser.add_argument("--stale-confidence-margin", type=float, default=0.08)
    parser.add_argument("--stale-guard-radius", type=int, default=1)
    parser.add_argument("--stale-inpaint-radius", type=float, default=3.0)
    parser.add_argument("--paint-erosion", type=int, default=1)
    parser.add_argument("--paint-kernel", choices=("square", "cross"), default="square")
    parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        parent, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str).tolist()
    # This checkpoint is produced locally by the paired training script and
    # contains NumPy scalar metric values in addition to tensor weights.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device); model = ContactOcclusionHeadV13(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.to(device).eval()
    probability = infer(model, checkpoint, parent, context, names, Path(args.windows), device)
    rendered, diagnostics = [], []
    for prediction, history, prob in zip(parent, context, probability):
        value, detail = render_sequence(prediction, history, prob, args.gripper_threshold,
                                        args.bottle_threshold, args.green_ratio, args.decay_ratio,
                                        args.recovery_ratio, args.material_ceiling,
                                        args.stale_confidence_margin, args.stale_guard_radius,
                                        args.stale_inpaint_radius, args.paint_erosion, args.paint_kernel)
        rendered.append(value); diagnostics.append(detail)
    rendered = np.stack(rendered)
    baseline, result = rgb_metrics(parent, target), rgb_metrics(rendered, target)
    report = {"format": "track2-contact-occlusion-head-v13-eval", "sample_count": len(parent),
              "checkpoint_step": int(checkpoint["step"]),
              "parameters": {"gripper_threshold": args.gripper_threshold,
                             "bottle_threshold": args.bottle_threshold, "green_ratio": args.green_ratio,
                             "decay_ratio": args.decay_ratio, "recovery_ratio": args.recovery_ratio,
                             "material_ceiling": args.material_ceiling,
                             "stale_confidence_margin": args.stale_confidence_margin,
                             "stale_guard_radius": args.stale_guard_radius,
                             "stale_inpaint_radius": args.stale_inpaint_radius,
                             "paint_erosion": args.paint_erosion,
                             "paint_kernel": args.paint_kernel},
              "learned_segmentation": learned_metrics(probability, target, args.gripper_threshold),
              "baseline": baseline, "rendered": result,
              "delta": {key: result[key] - baseline[key] for key in baseline}, "windows": diagnostics}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.output_cache:
        Path(args.output_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_cache, prediction=rendered, target=target, context=context,
                            windows=np.asarray(names), layer_probability=probability.astype(np.float16))
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
