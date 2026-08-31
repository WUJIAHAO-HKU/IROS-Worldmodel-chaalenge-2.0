#!/usr/bin/env python3
"""Create audited linear interpolations of two same-architecture AR checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import numpy as np


def npz_equal(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        return a.files == b.files and all(np.array_equal(a[key], b[key]) for key in a.files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--alphas", default="0.25,0.5,0.75")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    alphas = [float(value) for value in args.alphas.split(",")]
    if not alphas or any(not 0 < alpha < 1 for alpha in alphas):
        raise SystemExit("alphas must be strictly between zero and one")

    a, b = Path(args.checkpoint_a).resolve(), Path(args.checkpoint_b).resolve()
    state_a = torch.load(a / "model.pt", map_location="cpu", weights_only=True)
    state_b = torch.load(b / "model.pt", map_location="cpu", weights_only=True)
    if state_a.get("format") != "track2-autoregressive-unet-v1" or state_b.get("format") != state_a.get("format"):
        raise ValueError("both inputs must be Track 2 autoregressive checkpoints")
    if state_a["state_dict"].keys() != state_b["state_dict"].keys():
        raise ValueError("checkpoint parameter names differ")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in alphas:
        tag = f"alpha_{alpha:.4f}".replace(".", "p")
        output = output_root / tag
        if output.exists():
            raise FileExistsError(output)
        output.mkdir()
        interpolated = {}
        for key, value_a in state_a["state_dict"].items():
            value_b = state_b["state_dict"][key]
            if value_a.shape != value_b.shape or value_a.dtype != value_b.dtype:
                raise ValueError(f"parameter mismatch: {key}")
            if value_a.is_floating_point():
                interpolated[key] = value_a.lerp(value_b, alpha)
            else:
                interpolated[key] = value_a.clone()
        torch.save({"format": state_a["format"], "state_dict": interpolated}, output / "model.pt")
        for filename in ("action_normalization.npz", "track2_autoregressive_unet_config.npz"):
            if not npz_equal(a / filename, b / filename):
                raise ValueError(f"non-model artifact differs: {filename}")
            shutil.copy2(a / filename, output / filename)
        metadata = {
            "format": "track2-autoregressive-checkpoint-interpolation-v1",
            "checkpoint_a": str(a),
            "checkpoint_b": str(b),
            "alpha_b": alpha,
            "alpha_a": 1 - alpha,
            "purpose": "validation-only interpolation sweep; no validation labels enter weights",
        }
        (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append({"alpha": alpha, "checkpoint": str(output.resolve())})
    (output_root / "manifest.json").write_text(json.dumps({"format": "track2-ar-interpolation-sweep-v1", "rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
