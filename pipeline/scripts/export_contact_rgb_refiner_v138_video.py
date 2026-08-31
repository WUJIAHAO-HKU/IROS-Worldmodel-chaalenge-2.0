#!/usr/bin/env python3
"""Export all eight full-frame/contact-zoom comparisons for v13.8."""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def tile(frame, title, subtitle=""):
    canvas = Image.new("RGB", (384, 430), "white")
    canvas.paste(Image.fromarray(cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)), (0, 46))
    draw = ImageDraw.Draw(canvas); draw.text((8, 7), title, fill="black")
    if subtitle: draw.text((8, 25), subtitle, fill=(70, 70, 70))
    return np.asarray(canvas)


def zoom(parent, refined, target):
    canvas = Image.new("RGB", (384, 430), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "contact zoom: parent / v13.8 / truth", fill="black")
    for row, (frame, label) in enumerate(((parent, "parent"), (refined, "v13.8"), (target, "truth"))):
        crop = cv2.resize(frame[72:232, 30:210], (384, 120), interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(crop), (0, 46 + row * 128)); draw.text((5, 48 + row * 128), label, fill="red")
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--refined-cache", required=True); parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, target, context, names = cache["prediction"], cache["target"], cache["context"], cache["windows"].astype(str)
    with np.load(args.refined_cache, allow_pickle=False) as cache:
        refined, refined_names = cache["prediction"], cache["windows"].astype(str)
    if not np.array_equal(names, refined_names): raise ValueError("cache order differs")
    matches = np.flatnonzero(names == args.window)
    if len(matches) != 1: raise KeyError(args.window)
    index = int(matches[0]); frames = []
    for time in range(8):
        before = np.abs(parent[index, time].astype(float) - target[index, time]).mean()
        after = np.abs(refined[index, time].astype(float) - target[index, time]).mean()
        row = np.concatenate((tile(context[index, -1], "last observation"),
                              tile(target[index, time], f"ground truth t+{time + 1}"),
                              tile(parent[index, time], f"parent t+{time + 1}", f"MAE {before:.3f}"),
                              tile(refined[index, time], f"v13.8 RGB refiner t+{time + 1}", f"MAE {after:.3f}"),
                              zoom(parent[index, time], refined[index, time], target[index, time])), axis=1)
        frames.append(row)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    stem = f"v138_rgb_refiner_{Path(args.window).stem}"
    imageio.mimsave(output / f"{stem}.mp4", frames, fps=args.fps, codec="libx264", quality=8)
    imageio.mimsave(output / f"{stem}.gif", frames, fps=args.fps, loop=0)
    for time, frame in enumerate(frames): Image.fromarray(frame).save(output / f"frame_{time + 1:02d}.png")
    print(output / f"{stem}.mp4")


if __name__ == "__main__": main()
