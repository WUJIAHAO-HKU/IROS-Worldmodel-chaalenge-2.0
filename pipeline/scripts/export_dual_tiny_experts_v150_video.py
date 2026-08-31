#!/usr/bin/env python3
"""Export native-frame parent/expert/target comparisons for both active arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import REGIONS, _observed_logo_mask
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask


def label(frame: np.ndarray, text: str) -> np.ndarray:
    value = frame.copy()
    cv2.rectangle(value, (0, 0), (value.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(value, text, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (255, 255, 255), 1, cv2.LINE_AA)
    return value


def active_texture_mask(sequence: np.ndarray, arm: int) -> np.ndarray:
    side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
    output = np.zeros(sequence.shape[:3], bool)
    for time, frame in enumerate(sequence):
        local = _observed_logo_mask(frame[y0:y1, x0:x1])
        if local.any():
            output[time, y0:y1, x0:x1] = cv2.dilate(local, np.ones((5, 5), np.uint8)) > 0
    return output


def score(parent: np.ndarray, expert: np.ndarray, target: np.ndarray, arm: int) -> dict:
    error_parent = np.abs(parent.astype(np.float32) - target.astype(np.float32))
    error_expert = np.abs(expert.astype(np.float32) - target.astype(np.float32))
    texture = np.repeat(active_texture_mask(target, arm)[..., None], 3, axis=3)
    y0, y1, x0, x1 = CONTACT_REGION
    labels = structure_semantic_mask(target[:, y0:y1, x0:x1])
    structure = np.repeat(np.isin(labels, (2, 3))[..., None], 3, axis=3)
    parent_structure = error_parent[:, y0:y1, x0:x1][structure].mean() if structure.any() else np.nan
    expert_structure = error_expert[:, y0:y1, x0:x1][structure].mean() if structure.any() else np.nan
    parent_texture = error_parent[texture].mean() if texture.any() else np.nan
    expert_texture = error_expert[texture].mean() if texture.any() else np.nan
    gains = np.asarray((parent_structure - expert_structure,
                        parent_texture - expert_texture), np.float32)
    return {"rgb_parent": float(error_parent.mean()), "rgb_expert": float(error_expert.mean()),
            "structure_parent": float(parent_structure), "structure_expert": float(expert_structure),
            "texture_parent": float(parent_texture), "texture_expert": float(expert_texture),
            "combined_gain": float(np.nansum(gains))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True); parser.add_argument("--expert-cache", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--top-per-arm", type=int, default=2)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; parent_names = cache["windows"].astype(str)
    with np.load(args.expert_cache, allow_pickle=False) as cache:
        expert = cache["prediction"]; target = cache["target"]; context = cache["context"]
        names = cache["windows"].astype(str); arms = cache["arm_id"].astype(int)
    lookup = {name: index for index, name in enumerate(parent_names)}
    parent = np.stack([parent[lookup[name]] for name in names])
    scores = [score(parent[i], expert[i], target[i], int(arms[i])) for i in range(len(names))]
    selected = []
    for arm in (0, 1):
        candidates = [i for i in range(len(names)) if arms[i] == arm]
        candidates.sort(key=lambda index: scores[index]["combined_gain"], reverse=True)
        selected.extend(candidates[:args.top_per_arm])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); manifest = []
    for index in selected:
        video_frames = []
        for time in range(expert.shape[1]):
            tiles = [label(context[index, -1], "Observed t0"),
                     label(parent[index, time], f"Frozen parent t+{time + 1}"),
                     label(expert[index, time], f"Dual experts t+{time + 1}"),
                     label(target[index, time], f"Ground truth t+{time + 1}")]
            video_frames.append(np.concatenate(tiles, axis=1))
        stem = Path(names[index]).stem
        mp4 = output / f"dual_tiny_experts_v150_{stem}.mp4"
        writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                 (video_frames[0].shape[1], video_frames[0].shape[0]))
        for frame in video_frames: writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        gif = output / f"dual_tiny_experts_v150_{stem}.gif"
        images = [Image.fromarray(frame) for frame in video_frames]
        images[0].save(gif, save_all=True, append_images=images[1:],
                       duration=int(1000 / args.fps), loop=0)
        sheet = output / f"dual_tiny_experts_v150_{stem}_sheet.png"
        cv2.imwrite(str(sheet), cv2.cvtColor(np.concatenate(video_frames, axis=0), cv2.COLOR_RGB2BGR))
        manifest.append({"window": str(names[index]), "arm": int(arms[index]),
                         "scores": scores[index], "mp4": mp4.name,
                         "gif": gif.name, "sheet": sheet.name})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
