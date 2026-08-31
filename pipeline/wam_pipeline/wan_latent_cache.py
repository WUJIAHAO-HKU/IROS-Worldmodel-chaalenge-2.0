"""Validated on-disk latent caches for the frozen Track 2 Wan VAE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "track2-wan-latent-cache-v1"
SPLITS = ("train", "validation")


def sha256(path: str | Path) -> str:
    """Hash a small manifest or the immutable VAE weight file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(dataset_root: str | Path) -> dict[str, str]:
    root = Path(dataset_root).resolve()
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Track 2 latent cache needs dataset manifest: {manifest}")
    return {"dataset_root": str(root), "dataset_manifest_sha256": sha256(manifest)}


def vae_identity(base_model: str | Path) -> dict[str, str]:
    root = Path(base_model).resolve()
    config = root / "vae" / "config.json"
    weights = root / "vae" / "diffusion_pytorch_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError("Wan latent cache needs local vae/config.json and VAE safetensors weights")
    return {
        "base_model": str(root),
        "vae_config_sha256": sha256(config),
        "vae_weights_sha256": sha256(weights),
    }


def metadata_path(cache_root: str | Path) -> Path:
    return Path(cache_root) / "latent_cache_manifest.json"


def latent_path(cache_root: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown Track 2 latent-cache split: {split}")
    return Path(cache_root) / f"{split}_latents.npy"


def load_metadata(cache_root: str | Path) -> dict:
    path = metadata_path(cache_root)
    if not path.is_file():
        raise ValueError(f"missing Track 2 Wan latent-cache manifest: {path}")
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Track 2 Wan latent-cache manifest: {path}") from exc
    if metadata.get("format") != FORMAT:
        raise ValueError("unsupported Track 2 Wan latent-cache format")
    return metadata


def open_verified_cache(
    cache_root: str | Path,
    *,
    split: str,
    dataset_root: str | Path,
    base_model: str | Path,
    expected_windows: int,
    expected_channels: int,
    expected_latent_frames: int,
    expected_height: int,
    expected_width: int,
) -> np.ndarray:
    """Open a cache only when it is tied to these exact data and VAE weights."""
    metadata = load_metadata(cache_root)
    if metadata.get("dataset") != dataset_identity(dataset_root):
        raise ValueError("Track 2 Wan latent cache was built from different dataset bytes")
    if metadata.get("vae") != vae_identity(base_model):
        raise ValueError("Track 2 Wan latent cache was built with different VAE weights")
    split_info = metadata.get("splits", {}).get(split)
    if not isinstance(split_info, dict):
        raise ValueError(f"Track 2 Wan latent cache has no {split} split")
    expected_shape = [expected_windows, expected_channels, expected_latent_frames, expected_height, expected_width]
    if split_info.get("shape") != expected_shape or split_info.get("dtype") != "float16":
        raise ValueError(f"Track 2 Wan latent cache {split} shape/dtype does not match the active model")
    path = latent_path(cache_root, split)
    if not path.is_file():
        raise ValueError(f"missing Track 2 Wan latent cache file: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != expected_shape or values.dtype != np.float16:
        raise ValueError(f"Track 2 Wan latent cache file is malformed: {path}")
    return values
