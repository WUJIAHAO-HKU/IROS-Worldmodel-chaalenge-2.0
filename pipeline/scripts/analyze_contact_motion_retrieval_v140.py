#!/usr/bin/env python3
"""Audit action/appearance nearest-neighbour motion templates for one window."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from train_contact_occlusion_head_v131 import active_arm
from train_contact_occlusion_head_v13 import episode


def descriptor(frame):
    y0, y1, x0, x1 = CONTACT_REGION
    crop = cv2.resize(frame[y0:y1, x0:x1], (24, 20), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return crop.flatten()


def align(candidate_source, query_source):
    y0, y1, x0, x1 = CONTACT_REGION
    left = cv2.cvtColor(candidate_source[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    right = cv2.cvtColor(query_source[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(right, left, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                           None, 3)
    except cv2.error:
        score = -1.0
    return float(score), warp


def warp_crop(frames, warp):
    y0, y1, x0, x1 = CONTACT_REGION; result = []
    for frame in frames:
        crop = frame[y0:y1, x0:x1]
        result.append(cv2.warpAffine(crop, warp, (x1 - x0, y1 - y0),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT))
    return np.stack(result)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--windows", required=True)
    parser.add_argument("--validation-cache", required=True); parser.add_argument("--window", required=True)
    parser.add_argument("--top", type=int, default=20); args = parser.parse_args()
    windows = Path(args.windows)
    with np.load(args.validation_cache, allow_pickle=False) as cache:
        names = cache["windows"].astype(str); context = cache["context"]; target = cache["target"]
        parent = cache["prediction"]
    index = int(np.flatnonzero(names == args.window)[0]); query_source = context[index, -1]
    with np.load(windows / args.window, allow_pickle=False) as query:
        query_arm, query_action = active_arm(query["history_actions"], query["future_actions"])
    entries = []
    query_motion = np.concatenate(((query_action - query_action[:1]).reshape(-1),
                                   np.diff(query_action, axis=0).reshape(-1)))
    for path in sorted(windows.glob("*.npz")):
        if path.name in set(names) or episode(path.name) == episode(args.window): continue
        with np.load(path, allow_pickle=False) as candidate:
            arm, action = active_arm(candidate["history_actions"], candidate["future_actions"])
            if arm != query_arm: continue
            source = candidate["context_frames"][-1]
            visual = float(np.mean((descriptor(source) - descriptor(query_source)) ** 2))
            motion = np.concatenate(((action - action[:1]).reshape(-1),
                                     np.diff(action, axis=0).reshape(-1)))
            action_distance = float(np.mean((motion - query_motion) ** 2))
            entries.append((visual, action_distance, path.name))
    visual_scale = np.median([item[0] for item in entries]); action_scale = np.median([item[1] for item in entries])
    entries.sort(key=lambda item: item[0] / visual_scale + item[1] / action_scale)
    y0, y1, x0, x1 = CONTACT_REGION; truth = target[index, :, y0:y1, x0:x1]
    truth_label = structure_semantic_mask(truth); structure = np.isin(truth_label, (2, 3))[..., None]
    print("parent", np.abs(parent[index, :, y0:y1, x0:x1].astype(float) - truth).mean(),
          "structure", np.abs(parent[index, :, y0:y1, x0:x1].astype(float) - truth)[np.repeat(structure, 3, 3)].mean())
    for rank, (visual, action_distance, name) in enumerate(entries[:args.top], 1):
        with np.load(windows / name, allow_pickle=False) as candidate:
            source = candidate["context_frames"][-1]; future = candidate["target_frames"]
        ecc, warp = align(source, query_source); warped = warp_crop(future, warp)
        error = np.abs(warped.astype(float) - truth)
        print(rank, name, "visual", visual, "action", action_distance, "ecc", ecc,
              "mae", error.mean(), "structure", error[np.repeat(structure, 3, 3)].mean())


if __name__ == "__main__": main()
