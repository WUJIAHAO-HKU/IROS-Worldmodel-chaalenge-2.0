#!/usr/bin/env python3
"""Pre-encode 5+8 Track 2 windows with the deterministic SDXL VAE mean."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    root, output = Path(args.windows), Path(args.output)
    split = json.loads(Path(args.split_manifest).read_text())
    episodes = split[args.split.replace("-", "_") + "_episodes"]
    paths = [path for episode in episodes for path in sorted(root.glob(f"episode{episode}_*.npz"))]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no source windows")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    scaling = float(vae.config.scaling_factor)
    completed = 0
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(paths), args.batch_windows):
            group = paths[start : start + args.batch_windows]
            pending = [path for path in group if not (output / f"{path.stem}.npy").is_file()]
            if pending:
                videos = []
                for path in pending:
                    with np.load(path, allow_pickle=False) as data:
                        video = np.concatenate((data["context_frames"], data["target_frames"]), axis=0)
                    videos.append(video)
                value = torch.from_numpy(np.stack(videos)).to(device).permute(0, 1, 4, 2, 3).float().div(127.5).sub(1)
                batch, frames = value.shape[:2]
                latents = vae.encode(value.flatten(0, 1)).latent_dist.mode().mul(scaling).unflatten(0, (batch, frames))
                latents = latents.float().cpu().numpy().astype(np.float16)
                for path, latent in zip(pending, latents):
                    destination = output / f"{path.stem}.npy"
                    temporary = destination.with_suffix(f".npy.tmp.{os.getpid()}")
                    with temporary.open("wb") as handle:
                        np.save(handle, latent, allow_pickle=False)
                    os.replace(temporary, destination)
            completed += len(group)
            if completed % 100 == 0 or completed == len(paths):
                print(json.dumps({"completed": completed, "total": len(paths), "gpu_mb": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0}), flush=True)
    weights = Path(args.vae) / "vae" / "diffusion_pytorch_model.safetensors"
    manifest = {
        "format": "track2-sdxl-latent-windows-v1",
        "source_windows": str(root.resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "split": args.split,
        "window_count": len(paths),
        "frames": 13,
        "latent_shape": [13, 4, 32, 32],
        "dtype": "float16",
        "encoding": "posterior_mode",
        "vae": str(Path(args.vae).resolve()),
        "vae_sha256": sha256(weights),
        "scaling_factor": scaling,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
