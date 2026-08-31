#!/usr/bin/env python3
"""Mine a training-only canonical arm-texture atlas for v11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


REGIONS = {
    "left": (0, 128, 0, 128),
    "right": (0, 128, 128, 256),
}


def beam_support(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 2.5)
    dark = (blur < 0.43).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    support = np.zeros_like(dark)
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        center_y = centroids[index][1]
        if area >= 180 and width >= 34 and width >= 1.05 * height and center_y < 85:
            support[labels == index] = 1
    return cv2.dilate(support, np.ones((9, 9), np.uint8))


def logo_score(frame: np.ndarray, side: str) -> tuple[float, np.ndarray]:
    y0, y1, x0, x1 = REGIONS[side]
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    beam = beam_support(gray)
    bright_on_dark = ((gray > 0.48) & ((gray - blur) > 0.10) & (beam > 0)).astype(np.uint8)
    rgb = crop.astype(np.float32) / 255.0
    blue_logo = (
        (rgb[..., 2] > 0.28)
        & ((rgb[..., 2] - np.maximum(rgb[..., 0], rgb[..., 1])) > 0.07)
        & (beam > 0)
    ).astype(np.uint8)
    bright_on_dark[:12] = 0
    bright_on_dark[112:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright_on_dark, connectivity=8)
    accepted = np.zeros_like(bright_on_dark)
    component_score = 0.0
    component_count = 0
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if 2 <= area <= 90 and width <= 24 and height <= 18:
            accepted[labels == index] = 1
            component_score += area * (1.0 + min(width, height) / 8.0)
            component_count += 1
    blue_count, blue_labels, blue_stats, _ = cv2.connectedComponentsWithStats(blue_logo, connectivity=8)
    for index in range(1, blue_count):
        _, _, width, height, area = blue_stats[index]
        if 2 <= area <= 120 and width <= 28 and height <= 24:
            accepted[blue_labels == index] = 1
            component_score += 1.5 * area
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edge_score = float((laplacian * cv2.dilate(accepted, np.ones((5, 5), np.uint8))).sum())
    # Text is a collection of several small bright components on one dark beam.
    score = component_score + 7.5 * min(component_count, 12) + 2.0 * edge_score
    return float(score), accepted


def descriptor(frame: np.ndarray, side: str) -> np.ndarray:
    y0, y1, x0, x1 = REGIONS[side]
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    low = cv2.resize(cv2.GaussianBlur(gray, (0, 0), 4.0), (24, 24), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(low, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(low, cv2.CV_32F, 0, 1, ksize=3)
    value = np.stack((low, np.sqrt(gx * gx + gy * gy)), axis=0)
    value = value - value.mean(axis=(1, 2), keepdims=True)
    value = value / (value.std(axis=(1, 2), keepdims=True) + 1e-6)
    return value.astype(np.float16)


def labeled(frame: np.ndarray, title: str) -> np.ndarray:
    image = Image.new("RGB", (256, 284), "white")
    image.paste(Image.fromarray(frame), (0, 28))
    ImageDraw.Draw(image).text((5, 8), title, fill="black")
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--train-cache", required=True, help="Training parent cache; only its window names define the split.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--contact-count", type=int, default=40)
    parser.add_argument("--exclude-episodes", nargs="*", default=[])
    args = parser.parse_args()

    with np.load(args.train_cache, allow_pickle=False) as cache:
        names = cache["windows"].astype(str).tolist()
    excluded = set(args.exclude_episodes)
    names = [name for name in names if name.split("_")[0] not in excluded]
    records: list[tuple[float, str, int, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for window_index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            targets = window["target_frames"]
        for frame_index, frame in enumerate(targets):
            for side in REGIONS:
                score, text_mask = logo_score(frame, side)
                records.append((score, name, frame_index, side, frame.copy(), descriptor(frame, side), text_mask))
        if (window_index + 1) % 100 == 0:
            print(f"scanned {window_index + 1}/{len(names)} windows", flush=True)

    # Limit repeated adjacent frames so the atlas covers different poses and episodes.
    records.sort(key=lambda value: value[0], reverse=True)
    selected = []
    per_window_side: dict[tuple[str, str], int] = {}
    per_side = {side: 0 for side in REGIONS}
    side_quota = {side: args.top_k // len(REGIONS) for side in REGIONS}
    for side in list(REGIONS)[: args.top_k % len(REGIONS)]:
        side_quota[side] += 1
    for record in records:
        key = (record[1], record[3])
        if per_window_side.get(key, 0) >= 2:
            continue
        if per_side[record[3]] >= side_quota[record[3]]:
            continue
        selected.append(record)
        per_window_side[key] = per_window_side.get(key, 0) + 1
        per_side[record[3]] += 1
        if len(selected) == args.top_k:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=np.stack([record[4] for record in selected]),
        descriptors=np.stack([record[5] for record in selected]),
        text_masks=np.stack([record[6] for record in selected]),
        scores=np.asarray([record[0] for record in selected], dtype=np.float32),
        windows=np.asarray([record[1] for record in selected]),
        frame_indices=np.asarray([record[2] for record in selected], dtype=np.int16),
        sides=np.asarray([record[3] for record in selected]),
    )
    columns = 5
    previews = [
        labeled(record[4], f"{rank}: {record[1][:-4]} t{record[2] + 1} {record[3]} {record[0]:.1f}")
        for rank, record in enumerate(selected[: args.contact_count])
    ]
    rows = []
    blank = np.full_like(previews[0], 255)
    for start in range(0, len(previews), columns):
        row = previews[start : start + columns]
        rows.append(np.concatenate(row + [blank] * (columns - len(row)), axis=1))
    contact_path = output.with_name(output.stem + "_contact_sheet.png")
    Image.fromarray(np.concatenate(rows, axis=0)).save(contact_path)
    manifest = {
        "format": "track2-arm-texture-atlas-v11",
        "source": "training windows named by train-cache only",
        "excluded_episodes": sorted(excluded),
        "window_count": len(names),
        "candidate_frame_side_count": len(records),
        "atlas_size": len(selected),
        "regions": REGIONS,
        "top_score": selected[0][0],
        "minimum_score": selected[-1][0],
        "output": str(output.resolve()),
        "contact_sheet": str(contact_path.resolve()),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
