#!/usr/bin/env python3
"""Evaluate many AR checkpoints through one loaded frozen V15 composite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_autoregressive_candidate import Windows
from wam_pipeline.v15_gated_runtime import Track2V15GatedRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--candidate-release", required=True)
    parser.add_argument("--checkpoints", required=True, nargs="+")
    parser.add_argument("--reuse-baseline-cache", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mirror-checkpoint-to-right", action="store_true")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset = Windows(Path(args.windows), split["validation_episodes"])
    indices = list(range(len(dataset)))
    if args.max_windows and args.max_windows < len(indices):
        indices = np.linspace(0, len(dataset) - 1, args.max_windows, dtype=int).tolist()
    paths = [dataset.paths[index].name for index in indices]
    with np.load(args.reuse_baseline_cache, allow_pickle=False) as reused:
        if reused["path"].astype(str).tolist() != paths:
            raise ValueError("baseline cache does not contain the identical selected windows")
        common = {
            key: reused[key].copy()
            for key in (
                "context_last", "target", "baseline", "arm_right", "capture_success",
                "path", "synthetic_seed", "start",
            )
        }

    runtime = Track2V15GatedRuntime(args.candidate_release, args.library, args.device)
    expert = runtime.gated_autoregressive.candidate
    output_root = Path(args.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for checkpoint_value in args.checkpoints:
        checkpoint = Path(checkpoint_value).resolve()
        state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "track2-autoregressive-unet-v1":
            raise ValueError(f"unsupported checkpoint: {checkpoint}")
        expert.model.load_state_dict(state["state_dict"], strict=True)
        expert.model.eval()
        with np.load(checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
            expert.mean = normalization["mean"].astype(np.float32)
            expert.std = normalization["std"].astype(np.float32)
        if args.mirror_checkpoint_to_right:
            right_expert = runtime.gated_autoregressive.right_candidate
            if right_expert is None:
                raise RuntimeError("candidate release has no right expert")
            right_expert.model.load_state_dict(state["state_dict"], strict=True)
            right_expert.model.eval()
            right_expert.mean = expert.mean.copy()
            right_expert.std = expert.std.copy()
        predictions = []
        candidate_routes = 0
        for position, index in enumerate(indices, 1):
            context, history, future, target, _, _ = dataset[index]
            if not np.array_equal(common["context_last"][position - 1], context[-1]):
                raise ValueError(f"context mismatch: {dataset.paths[index].name}")
            if not np.array_equal(common["target"][position - 1], target):
                raise ValueError(f"target mismatch: {dataset.paths[index].name}")
            predictions.append(runtime.predict(context, history, future, 0, None))
            candidate_routes += int(
                str(runtime.gated_autoregressive.last_route).startswith("candidate")
            )
        tag = f"{checkpoint.parent.name}_{checkpoint.name}"
        output = output_root / f"{tag}.npz"
        # Store only the changing candidate tensor. The immutable GT/baseline
        # arrays live in reuse_baseline_cache and are verified by path identity.
        np.savez_compressed(
            output,
            candidate=np.stack(predictions),
            path=common["path"],
        )
        row = {
            "checkpoint": str(checkpoint),
            "output": str(output.resolve()),
            "window_count": len(indices),
            "candidate_routes": candidate_routes,
        }
        manifest_rows.append(row)
        print(json.dumps(row), flush=True)

    manifest = {
        "format": "strict-track2-reward-alignment-sweep-cache-v1",
        "candidate_release_shell": str(Path(args.candidate_release).resolve()),
        "reuse_baseline_cache": str(Path(args.reuse_baseline_cache).resolve()),
        "rows": manifest_rows,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
