#!/usr/bin/env python3
"""Precompute frozen-VAE latents for every exact Track 2 Wan window."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusers import AutoencoderKLWan

from wam_pipeline.track2_wan import TRACK2_LATENT_FRAMES, vae_latent_stats
from wam_pipeline.wan_latent_cache import (
    FORMAT,
    dataset_identity,
    latent_path,
    metadata_path,
    open_verified_cache,
    vae_identity,
)
from wam_pipeline.wan_track2_data import WanTrack2WindowDataset
from train_track2_wan import Track2WindowTorchDataset, encode_latents, frames_to_wan_video


def atomic_replace(source: Path, target: Path) -> None:
    source.replace(target)


def cache_split(
    split: str,
    *,
    dataset_root: Path,
    cache_root: Path,
    vae: AutoencoderKLWan,
    mean: torch.Tensor,
    inverse_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[int], str]:
    dataset = Track2WindowTorchDataset(dataset_root, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    spatial = 256 // int(vae.config.scale_factor_spatial)
    shape = (len(dataset), int(vae.config.z_dim), TRACK2_LATENT_FRAMES, spatial, spatial)
    target = latent_path(cache_root, split)
    temporary = target.with_suffix(".npy.tmp")
    temporary.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    # VAE outputs use BF16 on GPU; FP16 retains that frozen representation
    # while halving cache storage and transfer volume.
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    offset = 0
    for batch_index, (context, _, _, target_frames) in enumerate(loader, start=1):
        video = frames_to_wan_video(context, target_frames).to(device, non_blocking=True)
        latent = encode_latents(vae, video, mean, inverse_std).float().cpu().numpy()
        if not np.isfinite(latent).all():
            raise RuntimeError(f"non-finite latent in {split} batch {batch_index}")
        values[offset : offset + len(latent)] = latent
        offset += len(latent)
        if batch_index == 1 or batch_index % 50 == 0 or offset == len(dataset):
            print(json.dumps({"split": split, "cached_windows": offset, "total_windows": len(dataset)}), flush=True)
    values.flush()
    del values
    if offset != len(dataset):
        raise RuntimeError(f"latent-cache window count mismatch: {offset} != {len(dataset)}")
    atomic_replace(temporary, target)
    return list(shape), str(target.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen Wan2.2 VAE latents for exact Track 2 windows.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild an existing cache after explicit review.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("--batch-size must be positive and --num-workers must be non-negative")
    dataset_root, base_model, cache_root = Path(args.dataset_root), Path(args.base_model), Path(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    vae_config_path = base_model / "vae" / "config.json"
    if not vae_config_path.is_file():
        raise SystemExit("--base-model needs vae/config.json")
    vae_config = json.loads(vae_config_path.read_text())
    spatial = 256 // int(vae_config["scale_factor_spatial"])
    channels = int(vae_config["z_dim"])
    if metadata_path(cache_root).is_file() and not args.overwrite:
        try:
            for split in ("train", "validation"):
                open_verified_cache(
                    cache_root,
                    split=split,
                    dataset_root=dataset_root,
                    base_model=base_model,
                    expected_windows=len(WanTrack2WindowDataset(dataset_root, split)),
                    expected_channels=channels,
                    expected_latent_frames=TRACK2_LATENT_FRAMES,
                    expected_height=spatial,
                    expected_width=spatial,
                )
        except ValueError as exc:
            raise SystemExit(f"existing latent cache is not reusable: {exc}; use --overwrite to rebuild") from exc
        print(json.dumps({"status": "reused", "cache_root": str(cache_root.resolve())}, indent=2))
        return
    vae = AutoencoderKLWan.from_pretrained(
        str(base_model), subfolder="vae", torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    mean, inverse_std = vae_latent_stats(vae, device=device, dtype=next(vae.parameters()).dtype)
    splits: dict[str, dict] = {}
    for split in ("train", "validation"):
        shape, path = cache_split(
            split,
            dataset_root=dataset_root,
            cache_root=cache_root,
            vae=vae,
            mean=mean,
            inverse_std=inverse_std,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        splits[split] = {"shape": shape, "dtype": "float16", "path": path}
    metadata = {
        "format": FORMAT,
        "dataset": dataset_identity(dataset_root),
        "vae": vae_identity(base_model),
        "splits": splits,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path(cache_root).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, metadata_path(cache_root))
    print(json.dumps({"status": "complete", "cache_root": str(cache_root.resolve()), "splits": splits}, indent=2))


if __name__ == "__main__":
    main()
