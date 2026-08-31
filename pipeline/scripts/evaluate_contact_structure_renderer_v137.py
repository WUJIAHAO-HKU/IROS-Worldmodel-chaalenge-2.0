#!/usr/bin/env python3
"""Render explicit black/grey gripper geometry from the recurrent structure head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_masks, structure_semantic_mask


def pca_transform(source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray | None:
    source = np.argwhere(source_mask)[:, ::-1].astype(np.float64)
    target = np.argwhere(target_mask)[:, ::-1].astype(np.float64)
    if len(source) < 20 or len(target) < 20:
        return None
    source_center, target_center = source.mean(0), target.mean(0)
    source_cov = np.cov((source - source_center).T) + np.eye(2) * 1e-3
    target_cov = np.cov((target - target_center).T) + np.eye(2) * 1e-3
    source_values, source_vectors = np.linalg.eigh(source_cov)
    target_values, target_vectors = np.linalg.eigh(target_cov)
    source_order = np.argsort(source_values)[::-1]
    target_order = np.argsort(target_values)[::-1]
    source_values, source_vectors = source_values[source_order], source_vectors[:, source_order]
    target_values, target_vectors = target_values[target_order], target_vectors[:, target_order]
    for axis in range(2):
        if np.dot(source_vectors[:, axis], target_vectors[:, axis]) < 0:
            target_vectors[:, axis] *= -1
    scale = np.sqrt(np.clip(target_values / source_values, 0.40, 2.5))
    linear = target_vectors @ np.diag(scale) @ source_vectors.T
    translation = target_center - linear @ source_center
    return np.concatenate((linear, translation[:, None]), axis=1).astype(np.float32)


def material(source: np.ndarray, mask: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
    value = source.astype(np.float32)
    chroma = value.max(axis=2) - value.min(axis=2)
    mean = value.mean(axis=2)
    if label == 2:
        support = mask & (mean < 82) & (chroma < 42)
        fallback = np.percentile(value[support], 40, axis=0) if support.any() else np.asarray((38, 38, 38))
        fallback = np.minimum(fallback, 72).astype(np.float32)
    else:
        support = mask & (mean >= 82) & (mean < 205) & (chroma < 42)
        fallback = np.median(value[support], axis=0) if support.any() else np.asarray((142, 142, 142))
        fallback = np.clip(np.full(3, fallback.mean()), 88, 195).astype(np.float32)
    return support, fallback


def warp_material(source: np.ndarray, source_mask: np.ndarray, target_mask: np.ndarray,
                  label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support, fallback = material(source, source_mask, label)
    transform = pca_transform(source_mask, target_mask)
    if transform is None:
        texture = np.broadcast_to(fallback, source.shape).copy()
        warped_support = np.zeros(source_mask.shape, bool)
    else:
        size = (source.shape[1], source.shape[0])
        texture = cv2.warpAffine(source.astype(np.float32), transform, size,
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_support = cv2.warpAffine(support.astype(np.uint8), transform, size,
                                        flags=cv2.INTER_NEAREST) > 0
    if label == 2:
        texture = np.minimum(texture, 76)
    else:
        # Grey jaws must stay neutral. Chroma copied from a bottle boundary is
        # the source of the former black/green mixture.
        luminance = np.clip(texture.mean(axis=2), 88, 198)
        texture = np.repeat(luminance[..., None], 3, axis=2)
    return texture, warped_support, fallback


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.logical_and(left, right).sum() / max(np.logical_or(left, right).sum(), 1))


def should_render(parent: np.ndarray, source: np.ndarray, labels: np.ndarray,
                  probability: np.ndarray, minimum_decay: float,
                  minimum_agreement: float) -> tuple[bool, dict]:
    _, source_black, source_grey = structure_masks(source)
    source_structure = source_black | source_grey
    if source_black.sum() < 45 or source_grey.sum() < 45:
        return False, {"reason": "insufficient_observed_structure"}
    parent_labels = structure_semantic_mask(parent)
    parent_areas = np.isin(parent_labels, (2, 3)).sum(axis=(1, 2))
    ratios = parent_areas / max(int(source_structure.sum()), 1)
    predicted = np.isin(labels, (2, 3))
    agreement = mask_iou(predicted[0], np.isin(parent_labels[0], (2, 3)))
    confidence = float(np.take_along_axis(
        probability, labels[:, None], axis=1
    )[:, 0][predicted].mean()) if predicted.any() else 0.0
    accepted = bool(ratios.min() < minimum_decay and agreement >= minimum_agreement and confidence >= 0.55)
    reason = "accepted" if accepted else "gate_rejected"
    return accepted, {"reason": reason, "parent_area_ratios": ratios.tolist(),
                      "first_frame_agreement": agreement, "structure_confidence": confidence}


def render_sequence(parent: np.ndarray, context: np.ndarray, probability: np.ndarray,
                    minimum_decay: float = 0.82,
                    minimum_agreement: float = 0.58) -> tuple[np.ndarray, dict]:
    y0, y1, x0, x1 = CONTACT_REGION
    output = parent.copy()
    source = context[-1, y0:y1, x0:x1]
    labels = probability.argmax(axis=1)
    accepted, details = should_render(parent[:, y0:y1, x0:x1], source, labels,
                                      probability, minimum_decay, minimum_agreement)
    if not accepted:
        details["frames"] = []
        return output, details
    _, source_black, source_grey = structure_masks(source)
    frame_details = []
    for time in range(8):
        query_u8 = output[time, y0:y1, x0:x1].copy()
        _, old_black, old_grey = structure_masks(query_u8)
        black, grey = labels[time] == 2, labels[time] == 3
        predicted_structure = black | grey
        guard = cv2.dilate(predicted_structure.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        stale_core = (old_black | old_grey) & ~guard
        # One extra pixel removes the blurred black outline around the old pad.
        stale = cv2.dilate(stale_core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        stale &= ~predicted_structure
        cleaned = cv2.inpaint(query_u8, stale.astype(np.uint8) * 255, 3.0, cv2.INPAINT_TELEA)
        result = cleaned.astype(np.float32)
        for label, source_mask, target_mask in ((3, source_grey, grey), (2, source_black, black)):
            if target_mask.sum() < 15:
                continue
            texture, support, fallback = warp_material(source, source_mask, target_mask, label)
            values = np.where(support[..., None], texture, fallback)
            result[target_mask] = values[target_mask]
        output[time, y0:y1, x0:x1] = np.clip(np.round(result), 0, 255).astype(np.uint8)
        frame_details.append({"frame": time + 1, "black_pixels": int(black.sum()),
                              "grey_pixels": int(grey.sum()), "stale_pixels_removed": int(stale.sum())})
    details["frames"] = frame_details
    return output, details


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    y0, y1, x0, x1 = CONTACT_REGION
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    pred_labels = np.stack([structure_semantic_mask(value[:, y0:y1, x0:x1]) for value in prediction])
    target_labels = np.stack([structure_semantic_mask(value[:, y0:y1, x0:x1]) for value in target])
    result = {"rgb_mae": float(error.mean()),
              "contact_rgb_mae": float(error[:, :, y0:y1, x0:x1].mean())}
    for name, label in (("black", 2), ("grey", 3)):
        pred, truth = pred_labels == label, target_labels == label
        result[name + "_iou"] = mask_iou(pred, truth)
        result[name + "_precision"] = float(np.logical_and(pred, truth).sum() / max(pred.sum(), 1))
        result[name + "_recall"] = float(np.logical_and(pred, truth).sum() / max(truth.sum(), 1))
        result[name + "_frame_iou"] = [mask_iou(pred[:, time], truth[:, time]) for time in range(8)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--probability-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--minimum-decay", type=float, default=0.82)
    parser.add_argument("--minimum-agreement", type=float, default=0.58)
    args = parser.parse_args()
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.probability_cache, allow_pickle=False) as cache:
        probability = cache["probability"].astype(np.float32)
        probability_names = cache["windows"].astype(str)
        arms = cache["arm_id"]
    if not np.array_equal(names, probability_names):
        raise ValueError("cache window order differs")
    rendered, diagnostics = [], []
    for prediction, history, prob in zip(parent, context, probability):
        value, detail = render_sequence(prediction, history, prob,
                                        args.minimum_decay, args.minimum_agreement)
        rendered.append(value); diagnostics.append(detail)
    rendered = np.stack(rendered)
    report = {"format": "track2-contact-structure-renderer-v13.7",
              "sample_count": len(parent),
              "parameters": {"minimum_decay": args.minimum_decay,
                             "minimum_agreement": args.minimum_agreement},
              "accepted_count": int(sum(item["reason"] == "accepted" for item in diagnostics)),
              "overall": {"parent": metrics(parent, target), "rendered": metrics(rendered, target)},
              "arms": {}, "windows": []}
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm)
        report["arms"][f"arm{arm}"] = {"sample_count": len(indices),
                                          "parent": metrics(parent[indices], target[indices]),
                                          "rendered": metrics(rendered[indices], target[indices])}
    for index, name in enumerate(names):
        report["windows"].append({"window": name, "arm": int(arms[index]),
                                  "diagnostics": diagnostics[index],
                                  "parent": metrics(parent[index:index + 1], target[index:index + 1]),
                                  "rendered": metrics(rendered[index:index + 1], target[index:index + 1])})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    cache_output = Path(args.output_cache); cache_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_output, prediction=rendered, target=target, context=context,
                        windows=names, arm_id=arms)
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
