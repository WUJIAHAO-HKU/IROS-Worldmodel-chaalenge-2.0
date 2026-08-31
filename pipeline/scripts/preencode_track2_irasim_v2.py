#!/usr/bin/env python3
"""Cache native-aspect SDXL latents and Bridge-compatible physical actions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

from wam_pipeline.irasim_physical_actions import load_physical_window


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".npy.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation", "local-test"), required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-windows", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    windows = Path(args.windows)
    output = Path(args.output)
    latent_output, action_output = output / "latents", output / "actions"
    split = json.loads(Path(args.split_manifest).read_text())
    episodes = split[args.split.replace("-", "_") + "_episodes"]
    paths = [path for episode in episodes for path in sorted(windows.glob(f"episode{episode}_*.npz"))]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no source windows")

    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32).to(device).eval()
    scaling = float(vae.config.scaling_factor)
    arm_counts = {"left": 0, "right": 0}
    action_min = np.full(8, np.inf, dtype=np.float64)
    action_max = np.full(8, -np.inf, dtype=np.float64)
    completed = 0
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for start in range(0, len(paths), args.batch_windows):
            group = paths[start : start + args.batch_windows]
            physical = [load_physical_window(path) for path in group]
            for path, item in zip(group, physical):
                arm_counts[item.active_arm] += 1
                action_min = np.minimum(action_min, item.actions.min(0))
                action_max = np.maximum(action_max, item.actions.max(0))
                action_path = action_output / f"{path.stem}.npy"
                if not action_path.is_file():
                    _atomic_npy(action_path, item.actions.astype(np.float32))
            pending = [
                (path, item)
                for path, item in zip(group, physical)
                if not (latent_output / f"{path.stem}.npy").is_file()
            ]
            if pending:
                video = torch.from_numpy(np.stack([item.frames for _, item in pending])).to(device)
                video = video.permute(0, 1, 4, 2, 3).float().div(127.5).sub(1)
                batch, frames = video.shape[:2]
                latent = vae.encode(video.flatten(0, 1)).latent_dist.mode().mul(scaling)
                latent = latent.unflatten(0, (batch, frames)).float().cpu().numpy().astype(np.float16)
                for (path, _), value in zip(pending, latent):
                    _atomic_npy(latent_output / f"{path.stem}.npy", value)
            completed += len(group)
            if completed % 100 == 0 or completed == len(paths):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(paths),
                            "gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                        }
                    ),
                    flush=True,
                )
    manifest = {
        "format": "track2-irasim-v2-native-physical-cache-v1",
        "source_windows": str(windows.resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "split": args.split,
        "window_count": len(paths),
        "frames": 13,
        "frame_size_hw": [256, 320],
        "latent_shape": [13, 4, 32, 40],
        "action_shape": [12, 8],
        "action_semantics": "active-arm local EE delta xyz/rpy * 20, next gripper, arm_id(-1 left,+1 right)",
        "arm_counts": arm_counts,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "encoding": "SDXL posterior mode",
        "vae": str(Path(args.vae).resolve()),
        "scaling_factor": scaling,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
