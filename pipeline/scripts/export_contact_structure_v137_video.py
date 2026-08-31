#!/usr/bin/env python3
"""Export a full-frame and contact-zoom comparison for the v13.7 renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def tile(frame: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    canvas = Image.new("RGB", (384, 430), "white")
    canvas.paste(Image.fromarray(cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)), (0, 46))
    draw = ImageDraw.Draw(canvas); draw.text((8, 7), title, fill="black")
    if subtitle: draw.text((8, 25), subtitle, fill=(75, 75, 75))
    return np.asarray(canvas)


def zoom(parent: np.ndarray, rendered: np.ndarray, target: np.ndarray) -> np.ndarray:
    canvas = Image.new("RGB", (384, 430), "white")
    draw = ImageDraw.Draw(canvas); draw.text((8, 7), "contact zoom: parent / v13.7 / truth", fill="black")
    for row, (frame, label) in enumerate(((parent, "parent"), (rendered, "v13.7"), (target, "truth"))):
        crop = frame[82:226, 45:205]
        crop = cv2.resize(crop, (384, 120), interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(crop), (0, 46 + row * 128))
        draw.text((5, 48 + row * 128), label, fill=(255, 0, 0))
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--rendered-cache", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.rendered_cache, allow_pickle=False) as cache:
        rendered, rendered_names = cache["prediction"], cache["windows"].astype(str)
    if not np.array_equal(names, rendered_names): raise ValueError("cache window order differs")
    matches = np.flatnonzero(names == args.window)
    if len(matches) != 1: raise KeyError(args.window)
    index = int(matches[0]); frames = []
    for time in range(8):
        parent_mae = np.abs(parent[index, time].astype(float) - target[index, time]).mean()
        rendered_mae = np.abs(rendered[index, time].astype(float) - target[index, time]).mean()
        row = np.concatenate((
            tile(context[index, -1], "last observation"),
            tile(target[index, time], f"ground truth t+{time + 1}"),
            tile(parent[index, time], f"parent t+{time + 1}", f"RGB MAE {parent_mae:.3f}"),
            tile(rendered[index, time], f"v13.7 structure t+{time + 1}", f"RGB MAE {rendered_mae:.3f}"),
            zoom(parent[index, time], rendered[index, time], target[index, time]),
        ), axis=1)
        frames.append(row)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    stem = f"best_v137_full_structure_{Path(args.window).stem}"
    imageio.mimsave(output / f"{stem}.mp4", frames, fps=args.fps, codec="libx264", quality=8)
    imageio.mimsave(output / f"{stem}.gif", frames, fps=args.fps, loop=0)
    for time, frame in enumerate(frames): Image.fromarray(frame).save(output / f"frame_{time + 1:02d}.png")
    Image.fromarray(np.concatenate(frames, axis=0)).save(output / f"{stem}_sheet.png")
    metrics = json.loads(Path(args.metrics).read_text())
    window_metrics = next(item for item in metrics["windows"] if item["window"] == args.window)
    summary = {"format": "track2-v13.7-video", "window": args.window,
               "fps": args.fps, "frame_count": 8, "metrics": window_metrics}
    (output / f"{stem}.metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"mp4": str(output / f"{stem}.mp4"),
                      "gif": str(output / f"{stem}.gif")}, indent=2))


if __name__ == "__main__": main()
