#!/usr/bin/env python3
"""Calibrate a v10 risk router only on its episode-disjoint development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.candidate_risk_router_v10 import CandidateRiskRouterV10
from wam_pipeline.candidate_risk_router_v101 import CandidateRiskRouterV101
from train_candidate_risk_router_v10 import RiskDataset, evaluate, load_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--flow-head", required=True)
    parser.add_argument("--risk-checkpoint", required=True)
    parser.add_argument("--evaluation-samples", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.risk_checkpoint, map_location="cpu", weights_only=False)
    dev_episodes = {str(value) for value in checkpoint["dev_episodes"]}
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    indices = [index for index, name in enumerate(names) if name.split("_")[0] in dev_episodes]
    if args.evaluation_samples and len(indices) > args.evaluation_samples:
        positions = np.linspace(0, len(indices) - 1, args.evaluation_samples, dtype=np.int64)
        indices = [indices[int(position)] for position in positions]
    dataset = RiskDataset(Path(args.windows), parent, names, indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    base_checkpoint = Path(args.base_checkpoint)
    flow_model, active_mean, active_std = load_flow(base_checkpoint, Path(args.flow_head), device)
    with np.load(base_checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        action_mean = torch.from_numpy(normalization["mean"]).to(device)
        action_std = torch.from_numpy(normalization["std"]).to(device)
    router_version = str(checkpoint.get("router_version", "v10"))
    router_class = CandidateRiskRouterV101 if router_version == "v10.1" else CandidateRiskRouterV10
    router = router_class(int(checkpoint["base_channels"]))
    router.load_state_dict(checkpoint["state_dict"], strict=True)
    router = router.to(device).eval()
    metrics = evaluate(loader, flow_model, router, device, action_mean, action_std, active_mean, active_std)
    result = {
        "format": "track2-candidate-risk-router-v10-dev-calibration", "router_version": router_version,
        "risk_checkpoint": str(Path(args.risk_checkpoint).resolve()), "risk_step": int(checkpoint["step"]),
        "dev_episodes": sorted(dev_episodes), "sample_count": len(dataset), "metrics": metrics,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metrics.items() if not key.startswith("setting_")}, indent=2))


if __name__ == "__main__":
    main()
