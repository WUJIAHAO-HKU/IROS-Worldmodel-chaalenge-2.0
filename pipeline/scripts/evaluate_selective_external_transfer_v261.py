#!/usr/bin/env python3
"""Evaluate domain-safe transfer of deep motion features from v26 pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_autoregressive_unet import WindowDataset, evaluate, save_checkpoint
from wam_pipeline.autoregressive_unet import OneStepActionUNet
from wam_pipeline.external_robotwin_data_v260 import ExternalRandomizedWindowDataset, available_external_episodes


GROUPS = {
    "identity": (),
    "action": ("action_embedding.",),
    "bottleneck": ("action_embedding.", "enc4.", "middle."),
    "deep_motion": ("action_embedding.", "enc3.", "enc4.", "middle.", "up3."),
}


def load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    return state["state_dict"]


def normalization(checkpoint: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(checkpoint / "action_normalization.npz", allow_pickle=False) as data:
        mean = torch.from_numpy(np.asarray(data["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(data["std"], dtype=np.float32)).to(device)
    return mean, std


def blend(
    base: dict[str, torch.Tensor], donor: dict[str, torch.Tensor], prefixes: tuple[str, ...], alpha: float
) -> tuple[dict[str, torch.Tensor], list[str]]:
    selected = [key for key in base if any(key.startswith(prefix) for prefix in prefixes)]
    output = {key: value.clone() for key, value in base.items()}
    for key in selected:
        output[key] = base[key] + (donor[key] - base[key]) * alpha
    return output, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--official-windows", required=True)
    parser.add_argument("--official-split", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--external-validation-windows", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_path, donor_path = Path(args.base), Path(args.donor)
    base, donor = load_state(base_path), load_state(donor_path)
    if base.keys() != donor.keys() or any(base[key].shape != donor[key].shape for key in base):
        raise SystemExit("base and donor architectures do not match")
    mean, std = normalization(base_path, device)
    split = json.loads(Path(args.official_split).read_text())
    official = WindowDataset(Path(args.official_windows), split["validation_episodes"])
    official_loader = DataLoader(official, batch_size=args.batch_size, shuffle=False, num_workers=2)
    external_ids = available_external_episodes(args.external_root)
    external = ExternalRandomizedWindowDataset(
        args.external_root, episodes=set(external_ids[-50:]), stride=8
    )
    external_indices = np.linspace(
        0, len(external) - 1, min(len(external), args.external_validation_windows), dtype=np.int64
    ).tolist()
    external_loader = DataLoader(
        Subset(external, external_indices), batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    candidates = [("identity", 0.0)]
    candidates += [("action", alpha) for alpha in (0.25, 0.5, 1.0)]
    candidates += [("bottleneck", alpha) for alpha in (0.1, 0.25, 0.5)]
    candidates += [("deep_motion", alpha) for alpha in (0.1, 0.25, 0.5)]
    model = OneStepActionUNet().to(device)
    records = []
    output = Path(args.output)
    best_metric = float("inf")
    best_name = None
    for group, alpha in candidates:
        state, selected = blend(base, donor, GROUPS[group], alpha)
        model.load_state_dict(state, strict=True)
        official_result = evaluate(official_loader, model, device, mean, std, 0.04)
        external_result = evaluate(external_loader, model, device, mean, std, 0.04)
        name = f"{group}_a{alpha:g}"
        record = {
            "name": name,
            "group": group,
            "alpha": alpha,
            "transferred_tensor_count": len(selected),
            "official_dev": official_result,
            "external_dev": external_result,
        }
        records.append(record)
        metric = float(official_result["rollout_mae"])
        if metric < best_metric:
            best_metric, best_name = metric, name
            metadata = {
                "backend": "autoregressive_unet",
                "format": "track2-autoregressive-unet-v1",
                "stage": "v26.1-selective-external-feature-transfer",
                "base": str(base_path.resolve()),
                "donor": str(donor_path.resolve()),
                "candidate": record,
                "selection_metric": "full episode-disjoint official dev rollout MAE",
            }
            save_checkpoint(output / "best", model, mean, std, metadata)
        print(json.dumps({"candidate": name, "official": metric, "external": external_result["rollout_mae"]}), flush=True)
    baseline = float(records[0]["official_dev"]["rollout_mae"])
    report = {
        "format": "track2-v26.1-selective-external-transfer",
        "base": str(base_path.resolve()),
        "donor": str(donor_path.resolve()),
        "official_dev_windows": len(official),
        "external_dev_windows": len(external_indices),
        "baseline_official_dev_mae": baseline,
        "best_candidate": best_name,
        "best_official_dev_mae": best_metric,
        "best_improvement_percent": (baseline - best_metric) / baseline * 100.0,
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("best_candidate", "best_official_dev_mae", "best_improvement_percent")}, indent=2))


if __name__ == "__main__":
    main()
