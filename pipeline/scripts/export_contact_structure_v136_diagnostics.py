#!/usr/bin/env python3
"""Export all-frame contact-structure diagnostics for one evaluation window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask


COLOURS = {
    1: np.asarray((255, 210, 0), np.uint8),
    2: np.asarray((255, 30, 30), np.uint8),
    3: np.asarray((40, 170, 255), np.uint8),
}


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask & ~eroded


def overlay(frame: np.ndarray, labels: np.ndarray, alpha: float = 0.22) -> np.ndarray:
    result = frame.astype(np.float32).copy()
    y0, y1, x0, x1 = CONTACT_REGION
    crop = result[y0:y1, x0:x1]
    for label, colour in COLOURS.items():
        mask = labels == label
        crop[mask] = crop[mask] * (1.0 - alpha) + colour * alpha
        crop[boundary(mask)] = colour
    result[y0:y1, x0:x1] = crop
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def panel(frame: np.ndarray, labels: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    shown = overlay(frame, labels)
    canvas = Image.new("RGB", (384, 430), "white")
    canvas.paste(Image.fromarray(cv2.resize(shown, (384, 384), interpolation=cv2.INTER_NEAREST)), (0, 46))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill="black")
    if subtitle:
        draw.text((8, 25), subtitle, fill=(75, 75, 75))
    return np.asarray(canvas)


def class_metrics(prediction: np.ndarray, truth: np.ndarray, label: int) -> dict[str, float | int]:
    pred, target = prediction == label, truth == label
    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    return {
        "iou": float(intersection / max(union, 1)),
        "precision": float(intersection / max(int(pred.sum()), 1)),
        "recall": float(intersection / max(int(target.sum()), 1)),
        "predicted_pixels": int(pred.sum()),
        "target_pixels": int(target.sum()),
        "false_positive_pixels": int(np.logical_and(pred, ~target).sum()),
        "false_negative_pixels": int(np.logical_and(~pred, target).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--v135-cache", required=True)
    parser.add_argument("--v136-cache", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]
        target = cache["target"]
        context = cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.v135_cache, allow_pickle=False) as cache:
        probability135 = cache["probability"].astype(np.float32)
        names135 = cache["windows"].astype(str)
    with np.load(args.v136_cache, allow_pickle=False) as cache:
        probability136 = cache["probability"].astype(np.float32)
        names136 = cache["windows"].astype(str)
    if not np.array_equal(names, names135) or not np.array_equal(names, names136):
        raise ValueError("cache window order differs")
    matches = np.flatnonzero(names == args.window)
    if len(matches) != 1:
        raise KeyError(args.window)
    index = int(matches[0])
    y0, y1, x0, x1 = CONTACT_REGION
    truth = structure_semantic_mask(target[index, :, y0:y1, x0:x1])
    parent_label = structure_semantic_mask(parent[index, :, y0:y1, x0:x1])
    label135 = probability135[index].argmax(axis=1)
    label136 = probability136[index].argmax(axis=1)
    source_label = structure_semantic_mask(context[index, -1:, y0:y1, x0:x1])[0]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_panels, report_frames = [], []
    for time in range(8):
        metrics135 = {name: class_metrics(label135[time], truth[time], label)
                      for name, label in (("black", 2), ("grey", 3))}
        metrics136 = {name: class_metrics(label136[time], truth[time], label)
                      for name, label in (("black", 2), ("grey", 3))}
        panels = [
            panel(context[index, -1], source_label, "last observation", "red=black, blue=grey"),
            panel(target[index, time], truth[time], f"ground truth t+{time + 1}"),
            panel(parent[index, time], parent_label[time], f"parent t+{time + 1}"),
            panel(parent[index, time], label135[time], f"v13.5 mask t+{time + 1}",
                  f"B {metrics135['black']['iou']:.3f}  G {metrics135['grey']['iou']:.3f}"),
            panel(parent[index, time], label136[time], f"v13.6 motion mask t+{time + 1}",
                  f"B {metrics136['black']['iou']:.3f}  G {metrics136['grey']['iou']:.3f}"),
        ]
        row = np.concatenate(panels, axis=1)
        all_panels.append(row)
        Image.fromarray(row).save(output / f"frame_{time + 1:02d}.png")
        report_frames.append({"frame": time + 1, "v135": metrics135, "v136": metrics136})
    imageio.mimsave(output / "all_frames.mp4", all_panels, fps=args.fps, codec="libx264", quality=8)
    Image.fromarray(np.concatenate(all_panels, axis=0)).save(output / "all_frames_sheet.png")
    report = {"format": "track2-contact-structure-v13.6-all-frame-diagnostic",
              "window": args.window, "legend": {"red": "black structure", "blue": "grey structure",
                                                     "yellow": "bottle"}, "frames": report_frames}
    (output / "all_frames_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "frame_count": len(all_panels)}, indent=2))


if __name__ == "__main__":
    main()
