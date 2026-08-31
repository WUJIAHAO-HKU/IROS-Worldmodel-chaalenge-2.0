#!/usr/bin/env python3
"""Export a labeled v11 versus v12 contact-structure comparison video."""

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
    canvas.paste(Image.fromarray(frame), (0, 32))
    ImageDraw.Draw(canvas).text((6, 10), title, fill="black")
    return np.asarray(canvas)


def contact_zoom(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_crop = cv2.resize(before[92:222, 55:190], (256, 124), interpolation=cv2.INTER_NEAREST)
    after_crop = cv2.resize(after[92:222, 55:190], (256, 124), interpolation=cv2.INTER_NEAREST)
    canvas = Image.new("RGB", (256, 288), "white")
    canvas.paste(Image.fromarray(before_crop), (0, 32))
    canvas.paste(Image.fromarray(after_crop), (0, 164))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 10), "gripper-bottle contact zoom", fill="black")
    draw.text((5, 34), "v11", fill="red")
    draw.text((5, 166), "v12", fill="red")
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--v12-cache", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()
    with np.load(args.baseline_cache, allow_pickle=False) as cache:
        baseline, target, context = cache["prediction"], cache["target"], cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.v12_cache, allow_pickle=False) as cache:
        v12, v12_names = cache["prediction"], cache["windows"].astype(str)
    if not np.array_equal(names, v12_names):
        raise ValueError("baseline and v12 cache window order differs")
    matches = np.flatnonzero(names == args.window)
    if len(matches) != 1:
        raise KeyError(f"expected one {args.window}, found {len(matches)}")
    index = int(matches[0])
    frames = []
    for frame_index in range(target.shape[1]):
        base_mae = np.abs(baseline[index, frame_index].astype(float) - target[index, frame_index]).mean()
        v12_mae = np.abs(v12[index, frame_index].astype(float) - target[index, frame_index]).mean()
        frames.append(
            np.concatenate(
                (
                    tile(context[index, -1], "last observed frame"),
                    tile(target[index, frame_index], f"ground truth t+{frame_index + 1}"),
                    tile(baseline[index, frame_index], f"v11 MAE {base_mae:.3f}"),
                    tile(v12[index, frame_index], f"v12 contact MAE {v12_mae:.3f}"),
                    contact_zoom(baseline[index, frame_index], v12[index, frame_index]),
                ),
                axis=1,
            )
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"best_v12_contact_layer_{Path(args.window).stem}"
    imageio.mimsave(output_dir / f"{stem}.gif", frames, fps=args.fps, loop=0)
    imageio.mimsave(output_dir / f"{stem}.mp4", frames, fps=args.fps, codec="libx264", quality=8)
    Image.fromarray(np.concatenate(frames, axis=0)).save(output_dir / f"{stem}_contact_sheet.png")
    source_metrics = json.loads(Path(args.metrics).read_text())
    video_metrics = {
        "format": "track2-contact-layer-v12-video",
        "window": args.window,
        "fps": args.fps,
        "frame_count": len(frames),
        "width": int(frames[0].shape[1]),
        "height": int(frames[0].shape[0]),
        "evaluation": {key: value for key, value in source_metrics.items() if key != "windows"},
    }
    (output_dir / f"{stem}.metrics.json").write_text(json.dumps(video_metrics, indent=2) + "\n")
    print(f"wrote {stem} MP4/GIF/contact sheet with {len(frames)} frames")


if __name__ == "__main__":
    main()

