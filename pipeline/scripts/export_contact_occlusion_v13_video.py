#!/usr/bin/env python3
"""Export a labeled parent/v13/target contact comparison video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def tile(frame: np.ndarray, title: str) -> np.ndarray:
    canvas = Image.new("RGB", (256, 288), "white")
    canvas.paste(Image.fromarray(frame), (0, 32)); ImageDraw.Draw(canvas).text((6, 10), title, fill="black")
    return np.asarray(canvas)


def zoom(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before = cv2.resize(before[92:222, 55:190], (256, 124), interpolation=cv2.INTER_NEAREST)
    after = cv2.resize(after[92:222, 55:190], (256, 124), interpolation=cv2.INTER_NEAREST)
    canvas = Image.new("RGB", (256, 288), "white"); canvas.paste(Image.fromarray(before), (0, 32))
    canvas.paste(Image.fromarray(after), (0, 164)); draw = ImageDraw.Draw(canvas)
    draw.text((6, 10), "contact zoom: parent / recurrent v13", fill="black")
    draw.text((5, 34), "parent", fill="red"); draw.text((5, 166), "v13", fill="red")
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", required=True); parser.add_argument("--v13-cache", required=True)
    parser.add_argument("--metrics", required=True); parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()
    with np.load(args.baseline_cache, allow_pickle=False) as cache:
        baseline, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.v13_cache, allow_pickle=False) as cache:
        v13, v13_names = cache["prediction"], cache["windows"].astype(str)
    if not np.array_equal(names, v13_names): raise ValueError("cache window order differs")
    matches = np.flatnonzero(names == args.window)
    if len(matches) != 1: raise KeyError(args.window)
    index = int(matches[0]); frames = []
    for time in range(8):
        base_mae = np.abs(baseline[index, time].astype(float) - target[index, time]).mean()
        v13_mae = np.abs(v13[index, time].astype(float) - target[index, time]).mean()
        frames.append(np.concatenate((tile(context[index, -1], "last observation"),
                                      tile(target[index, time], f"ground truth t+{time + 1}"),
                                      tile(baseline[index, time], f"parent MAE {base_mae:.3f}"),
                                      tile(v13[index, time], f"recurrent v13 MAE {v13_mae:.3f}"),
                                      zoom(baseline[index, time], v13[index, time])), axis=1))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    stem = f"best_v13_recurrent_contact_{Path(args.window).stem}"
    imageio.mimsave(output / f"{stem}.gif", frames, fps=args.fps, loop=0)
    imageio.mimsave(output / f"{stem}.mp4", frames, fps=args.fps, codec="libx264", quality=8)
    Image.fromarray(np.concatenate(frames, axis=0)).save(output / f"{stem}_contact_sheet.png")
    metrics = json.loads(Path(args.metrics).read_text()); summary = {"format": "track2-v13-video",
        "window": args.window, "fps": args.fps, "frame_count": 8,
        "evaluation": {key: value for key, value in metrics.items() if key != "windows"}}
    (output / f"{stem}.metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"mp4": str(output / f"{stem}.mp4"), "gif": str(output / f"{stem}.gif")}))


if __name__ == "__main__": main()
