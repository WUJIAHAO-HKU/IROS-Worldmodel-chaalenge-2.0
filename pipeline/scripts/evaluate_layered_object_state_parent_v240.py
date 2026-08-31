#!/usr/bin/env python3
"""Calibrate v24's sparse deployment gate on episode-disjoint development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.layered_object_state_parent_v240 import LayeredObjectStateParentV240
from train_layered_object_state_parent_v240 import action_stats, episode, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True); parser.add_argument("--precompute-cache", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--dev-episodes", default="36,47")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[.5, .7, .8, .9, .95, .98, .99])
    parser.add_argument("--maximum-windows", type=int, default=0); parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--active-from", nargs="+", type=int, default=[0])
    parser.add_argument("--beam-profiles", nargs="+", choices=("on", "off"), default=["on"])
    args = parser.parse_args(); device = torch.device("cuda")
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; target = cache["target"]; context = cache["context"]
        names = cache["windows"].astype(str)
    with np.load(args.precompute_cache, allow_pickle=False) as cache:
        if not np.array_equal(cache["windows"].astype(str), names):
            raise ValueError("precompute cache mismatch")
        source = cache["source_geometry"]; future = cache["future_geometry"]
        arms = cache["arms"]; actions = cache["actions"]
    dev_episodes = {int(value) for value in args.dev_episodes.split(",") if value}
    train = [i for i, name in enumerate(names) if episode(name) not in dev_episodes]
    dev = [i for i, name in enumerate(names) if episode(name) in dev_episodes]
    stats = action_stats(actions, arms, train)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LayeredObjectStateParentV240(
        int(checkpoint["channels"]), float(checkpoint["maximum_residual_flow"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval().requires_grad_(False)
    results = []
    for threshold in args.thresholds:
        for active_from in args.active_from:
            for beam_profile in args.beam_profiles:
                enabled = (beam_profile == "on", True, True, True)
                metrics = evaluate(
                    model, dev, (parent, target, context), (source, future), actions,
                    arms, stats, device, args.maximum_windows, args.workers, threshold,
                    active_from, enabled,
                )
                metrics.update({"mask_threshold": threshold, "active_from": active_from,
                                "beam_profile": beam_profile}); results.append(metrics)
                print(json.dumps({"threshold": threshold, "active_from": active_from,
                                  "beam_profile": beam_profile, "metrics": metrics}), flush=True)
    results.sort(key=lambda value: value["gate_score"], reverse=True)
    report = {"format": "track2-v24.1-dev-mask-threshold-sweep", "checkpoint": args.checkpoint,
              "checkpoint_step": checkpoint["step"], "dev_episodes": sorted(dev_episodes),
              "windows": len(dev) if not args.maximum_windows else min(len(dev), args.maximum_windows),
              "results": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
