#!/usr/bin/env python3
"""Export a labeled v10.1 versus v11 canonical-texture comparison video."""

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


def zoom_tile(v101: np.ndarray, v11: np.ndarray) -> np.ndarray:
    # Fixed-camera right-arm beam crop, enlarged for letter inspection.
    crop_v101 = cv2.resize(v101[22:112, 136:256], (256, 124), interpolation=cv2.INTER_NEAREST)
    crop_v11 = cv2.resize(v11[22:112, 136:256], (256, 124), interpolation=cv2.INTER_NEAREST)
    canvas = Image.new("RGB", (256, 288), "white")
    canvas.paste(Image.fromarray(crop_v101), (0, 32))
    canvas.paste(Image.fromarray(crop_v11), (0, 164))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 10), "right-arm text zoom", fill="black")
    draw.text((5, 34), "v10.1", fill="red")
    draw.text((5, 166), "v11", fill="red")
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--v11-cache", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()

    with np.load(args.baseline_cache, allow_pickle=False) as cache:
        baseline = cache["prediction"]
        target = cache["target"]
        context = cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.v11_cache, allow_pickle=False) as cache:
        v11 = cache["prediction"]
        v11_names = cache["windows"].astype(str)
    if not np.array_equal(names, v11_names):
        raise ValueError("baseline and v11 cache window order differs")
    matches = np.flatnonzero(names == args.window)
    if len(matches) != 1:
        raise KeyError(f"expected one {args.window}, found {len(matches)}")
    index = int(matches[0])
    frames = []
    for frame_index in range(target.shape[1]):
        base_mae = float(np.abs(baseline[index, frame_index].astype(float) - target[index, frame_index]).mean())
        v11_mae = float(np.abs(v11[index, frame_index].astype(float) - target[index, frame_index]).mean())
        frames.append(
            np.concatenate(
                (
                    tile(context[index, -1], "last observed frame"),
                    tile(target[index, frame_index], f"ground truth t+{frame_index + 1}"),
                    tile(baseline[index, frame_index], f"v10.1 MAE {base_mae:.3f}"),
                    tile(v11[index, frame_index], f"v11 safe MAE {v11_mae:.3f}"),
                    zoom_tile(baseline[index, frame_index], v11[index, frame_index]),
                ),
                axis=1,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"best_v11_canonical_texture_{Path(args.window).stem}"
    imageio.mimsave(output_dir / f"{stem}.gif", frames, fps=args.fps, loop=0)
    imageio.mimsave(output_dir / f"{stem}.mp4", frames, fps=args.fps, codec="libx264", quality=8)
    Image.fromarray(np.concatenate(frames, axis=0)).save(output_dir / f"{stem}_contact_sheet.png")
    source_metrics = json.loads(Path(args.metrics).read_text())
    video_metrics = {
        "format": "track2-canonical-arm-texture-v11-video",
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
