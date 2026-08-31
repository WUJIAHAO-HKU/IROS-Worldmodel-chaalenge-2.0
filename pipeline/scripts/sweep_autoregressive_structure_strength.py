#!/usr/bin/env python3
"""Select a safe explicit dark-structure coupling on held-out windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_autoregressive_structure_unet import WindowDataset, evaluate, selection_score
from wam_pipeline.autoregressive_structure_unet import OneStepActionSeparatedStructureUNet, OneStepActionStructureUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--strengths", default="0,0.0025,0.005,0.01,0.02,0.04")
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset = WindowDataset(Path(args.windows), split["validation_episodes"])
    count = min(len(dataset), args.samples)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    checkpoint = Path(args.checkpoint_dir)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") not in {"track2-autoregressive-structure-unet-v2", "track2-autoregressive-structure-unet-v3"}:
        raise ValueError("strength sweep requires a coupled structure checkpoint")
    normalization = np.load(checkpoint / "action_normalization.npz", allow_pickle=False)
    device = torch.device(args.device)
    mean = torch.from_numpy(np.asarray(normalization["mean"], np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], np.float32)).to(device)
    model_class = OneStepActionSeparatedStructureUNet if state.get("format") == "track2-autoregressive-structure-unet-v3" else OneStepActionStructureUNet
    model = model_class().to(device).eval()
    model.load_state_dict(state["state_dict"], strict=True)
    strengths = [float(value) for value in args.strengths.split(",")]
    records = []
    baseline = None
    for strength in strengths:
        model.structure_strength.data.fill_(strength)
        metrics = evaluate(loader, model, device, mean, std, args.high_motion_threshold, use_autocast=False)
        if baseline is None:
            if strength != 0.0:
                raise ValueError("the first strength must be zero")
            baseline = metrics
        record = {"strength": strength, "selection_score": selection_score(baseline, metrics), **metrics}
        records.append(record)
        print(json.dumps(record), flush=True)
    assert baseline is not None
    result = {
        "format": "track2-autoregressive-structure-strength-sweep-v1",
        "checkpoint_dir": str(checkpoint.resolve()),
        "sample_count": count,
        "strengths": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sample_count": count}, indent=2))


if __name__ == "__main__":
    main()
