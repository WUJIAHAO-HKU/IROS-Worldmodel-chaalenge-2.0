#!/usr/bin/env python3
"""Export all eight full-frame/contact-zoom comparisons for v14.0 retrieval."""

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


def zoom(parent, refined, target, version_label):
    canvas = Image.new("RGB", (384, 430), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"contact zoom: parent / {version_label} / truth", fill="black")
    for row, (frame, label) in enumerate(((parent, "parent"), (refined, version_label), (target, "truth"))):
        crop = cv2.resize(frame[72:232, 30:210], (384, 120), interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(crop), (0, 46 + row * 128)); draw.text((5, 48 + row * 128), label, fill="red")
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--retrieval-cache", required=True); parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--version-label", default="v14.0")
    args = parser.parse_args()
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, target, context, names = cache["prediction"], cache["target"], cache["context"], cache["windows"].astype(str)
    with np.load(args.retrieval_cache, allow_pickle=False) as cache:
        refined, refined_names = cache["prediction"], cache["windows"].astype(str)
    matches = np.flatnonzero(names == args.window)
    refined_matches = np.flatnonzero(refined_names == args.window)
    if len(matches) != 1 or len(refined_matches) != 1: raise KeyError(args.window)
    index = int(matches[0]); refined_index = int(refined_matches[0]); frames = []
    for time in range(8):
        before = np.abs(parent[index, time].astype(float) - target[index, time]).mean()
        after = np.abs(refined[refined_index, time].astype(float) - target[index, time]).mean()
        row = np.concatenate((tile(context[index, -1], "last observation"),
                              tile(target[index, time], f"ground truth t+{time + 1}"),
                              tile(parent[index, time], f"parent t+{time + 1}", f"MAE {before:.3f}"),
                              tile(refined[refined_index, time], f"{args.version_label} retrieval t+{time + 1}", f"MAE {after:.3f}"),
                              zoom(parent[index, time], refined[refined_index, time], target[index, time], args.version_label)), axis=1)
        frames.append(row)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    version_slug = args.version_label.lower().replace(".", "").replace(" ", "_")
    stem = f"best_{version_slug}_motion_retrieval_{Path(args.window).stem}"
    imageio.mimsave(output / f"{stem}.mp4", frames, fps=args.fps, codec="libx264", quality=8)
    imageio.mimsave(output / f"{stem}.gif", frames, fps=args.fps, loop=0)
    for time, frame in enumerate(frames): Image.fromarray(frame).save(output / f"frame_{time + 1:02d}.png")
    Image.fromarray(np.concatenate(frames, axis=0)).save(output / f"{stem}_sheet.png")
    print(output / f"{stem}.mp4")


if __name__ == "__main__": main()
