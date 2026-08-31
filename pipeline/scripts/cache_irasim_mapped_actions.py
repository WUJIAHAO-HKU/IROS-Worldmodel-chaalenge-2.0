#!/usr/bin/env python3
"""Cache production-valid Bridge actions predicted only from API joint commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.irasim_action_mapper import JointActionMapper, active_arm_identity, joint_action_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation", "local-test"), required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.mapper, map_location="cpu", weights_only=False)
    model = JointActionMapper()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    split = json.loads(Path(args.split_manifest).read_text())
    episodes = split[args.split.replace("-", "_") + "_episodes"]
    windows = Path(args.windows)
    paths = [path for episode in episodes for path in sorted(windows.glob(f"episode{episode}_*.npz"))]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            destination = output / f"{path.stem}.npy"
            if destination.is_file():
                continue
            with np.load(path, allow_pickle=False) as item:
                actions = np.concatenate((item["history_actions"], item["future_actions"])).astype(np.float32)
            features = torch.from_numpy(joint_action_features(actions)).to(args.device)
            bridge = model(features).cpu().numpy().astype(np.float32)
            arm_id = np.full((len(bridge), 1), active_arm_identity(actions), dtype=np.float32)
            bridge = np.concatenate((bridge, arm_id), axis=1)
            temporary = destination.with_suffix(f".npy.tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                np.save(handle, bridge, allow_pickle=False)
            os.replace(temporary, destination)
            if index % 500 == 0 or index == len(paths):
                print(json.dumps({"completed": index, "total": len(paths)}), flush=True)
    (output / "manifest.json").write_text(
        json.dumps({"format": "track2-irasim-mapped-actions-v1", "split": args.split, "windows": len(paths), "mapper": str(Path(args.mapper).resolve())}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
