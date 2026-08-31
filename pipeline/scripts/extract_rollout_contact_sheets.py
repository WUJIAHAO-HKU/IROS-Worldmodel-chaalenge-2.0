#!/usr/bin/env python3
"""Extract evenly spaced diagnostic contact sheets from rollout videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    last_frames = []
    for path in sorted(args.videos.glob("*.mp4")):
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for index in np.linspace(0, max(count - 1, 0), args.samples).round().astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode {path} frame {index}")
            cv2.putText(frame, f"{path.stem}:f{index}/{count}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            frames.append(frame)
        capture.release()
        sheet = np.concatenate(frames, axis=1)
        if not cv2.imwrite(str(args.output / f"{path.stem}_sheet.jpg"), sheet):
            raise RuntimeError("failed to write contact sheet")
        last_frames.append(frames[-1])
    if last_frames:
        if not cv2.imwrite(str(args.output / "all_last.jpg"), np.concatenate(last_frames, axis=1)):
            raise RuntimeError("failed to write last-frame sheet")


if __name__ == "__main__":
    main()
