#!/usr/bin/env python3
"""Extract RGB-free future gripper geometry supervision from external episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from train_contact_occlusion_head_v131 import active_arm
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.external_robotwin_data_v260 import ExternalRandomizedWindowDataset


SIZE = 96


def resize_rgb(value: np.ndarray) -> np.ndarray:
    return cv2.resize(value, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def resize_mask(value: np.ndarray) -> np.ndarray:
    if value.ndim == 2:
        return cv2.resize(value.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return np.stack([resize_mask(frame) for frame in value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--minimum-structure-area", type=int, default=400)
    parser.add_argument("--minimum-changed-pixels", type=int, default=120)
    parser.add_argument("--max-per-arm", type=int, default=2200)
    args = parser.parse_args()
    dataset = ExternalRandomizedWindowDataset(args.external_root, stride=args.stride)
    records = {0: [], 1: []}
    y0, y1, x0, x1 = CONTACT_REGION
    for index in range(len(dataset)):
        context, history, future, target = dataset[index]
        context, history, future, target = (
            context.numpy(), history.numpy(), future.numpy(), target.numpy()
        )
        source = context[-1, y0:y1, x0:x1]
        target_crop = target[:, y0:y1, x0:x1]
        source_label = structure_semantic_mask(source[None])[0]
        target_label = structure_semantic_mask(target_crop)
        source_structure = np.isin(source_label, (2, 3))
        target_structure = np.isin(target_label, (2, 3))
        previous = np.concatenate((source_structure[None], target_structure[:-1]), axis=0)
        changed = np.logical_xor(target_structure, previous)
        if target_structure.sum(1).sum(1).mean() < args.minimum_structure_area:
            continue
        if changed.sum() < args.minimum_changed_pixels:
            continue
        arm, action = active_arm(history, future)
        if len(records[arm]) >= args.max_per_arm:
            continue
        records[arm].append({
            "source_rgb": resize_rgb(source),
            "source_structure": resize_mask(source_structure),
            "source_bottle": resize_mask(source_label == 1),
            "target_structure": resize_mask(target_structure),
            "action": action.astype(np.float32),
            "episode": int(dataset.entries[index][0].parent.name.split("_")[-1]),
            "start": int(dataset.entries[index][1]),
        })
        if (index + 1) % 250 == 0:
            print(json.dumps({"scanned": index + 1, "selected_arm0": len(records[0]),
                              "selected_arm1": len(records[1])}), flush=True)
        if all(len(records[arm]) >= args.max_per_arm for arm in (0, 1)):
            break
    merged = records[0] + records[1]
    if not merged:
        raise RuntimeError("no external structure windows passed selection")
    order = np.random.default_rng(266).permutation(len(merged))
    merged = [merged[int(index)] for index in order]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["format"] = "track2-v26.6-external-structure-geometry"
        handle.attrs["source"] = str(Path(args.external_root).resolve())
        handle.create_dataset("source_rgb", data=np.stack([x["source_rgb"] for x in merged]),
                              compression="gzip", compression_opts=4, chunks=(1, SIZE, SIZE, 3))
        handle.create_dataset("source_structure", data=np.stack([x["source_structure"] for x in merged]),
                              compression="gzip", compression_opts=4, chunks=(8, SIZE, SIZE))
        handle.create_dataset("source_bottle", data=np.stack([x["source_bottle"] for x in merged]),
                              compression="gzip", compression_opts=4, chunks=(8, SIZE, SIZE))
        handle.create_dataset("target_structure", data=np.stack([x["target_structure"] for x in merged]),
                              compression="gzip", compression_opts=4, chunks=(1, 8, SIZE, SIZE))
        handle.create_dataset("action", data=np.stack([x["action"] for x in merged]))
        handle.create_dataset("arm", data=np.asarray([0] * len(records[0]) + [1] * len(records[1]))[order])
        handle.create_dataset("episode", data=np.asarray([x["episode"] for x in merged], np.int32))
        handle.create_dataset("start", data=np.asarray([x["start"] for x in merged], np.int32))
    report = {
        "format": "track2-v26.6-external-structure-geometry-cache",
        "external_windows_scanned": len(dataset), "selected_windows": len(merged),
        "arm0_windows": len(records[0]), "arm1_windows": len(records[1]),
        "minimum_structure_area": args.minimum_structure_area,
        "minimum_changed_pixels": args.minimum_changed_pixels,
        "rgb_supervision": False, "target_fields": ["combined_black_gray_structure_mask"],
        "output": str(output.resolve()), "bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
