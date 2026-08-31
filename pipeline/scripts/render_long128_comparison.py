#!/usr/bin/env python3
"""Render right-arm long-horizon target/baseline/candidate frame strips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def image_from_array(value: np.ndarray) -> Image.Image:
    array = np.asarray(value)
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        if float(array.max()) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=4)
    args = parser.parse_args()
    with np.load(args.baseline_cache, allow_pickle=False) as values:
        target = values["target"].copy()
        baseline = values["baseline"].copy()
        right = values["arm_right"].astype(bool)
        success = values["capture_success"].astype(bool)
        paths = values["path"].astype(str)
    with np.load(args.candidate_cache, allow_pickle=False) as values:
        candidate = values["candidate"].copy()
        candidate_paths = values["path"].astype(str)
    if not np.array_equal(paths, candidate_paths):
        raise ValueError("cache paths do not align")
    indices = np.flatnonzero(right & success)[: args.max_sequences]
    if not len(indices):
        raise ValueError("no right-arm success sequences")
    frames = [0, 31, 63, 95, 127]
    labels = ("target", "baseline", "candidate")
    sources = (target, baseline, candidate)
    tile = image_from_array(target[indices[0], 0])
    width, height = tile.size
    label_width = 105
    header_height = 24
    row_gap = 5
    sequence_height = header_height + len(labels) * (height + row_gap)
    canvas = Image.new(
        "RGB", (label_width + len(frames) * width, len(indices) * sequence_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for sequence_row, index in enumerate(indices):
        y0 = sequence_row * sequence_height
        draw.text((4, y0 + 4), f"{paths[index]}", fill="black")
        for column, frame in enumerate(frames):
            draw.text((label_width + column * width + 4, y0 + 4), f"t={frame}", fill="black")
        for source_row, (label, source) in enumerate(zip(labels, sources)):
            y = y0 + header_height + source_row * (height + row_gap)
            draw.text((4, y + height // 2), label, fill="black")
            for column, frame in enumerate(frames):
                canvas.paste(image_from_array(source[index, frame]), (label_width + column * width, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
