#!/usr/bin/env python3
"""Evaluate Cartesian monotonic phase alignment on untouched public holdouts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.data import load_robotwin_hdf5
from wam_pipeline.object_geometry_v170 import ActionPoseProjector, denormalize_pose, normalize_action
from wam_pipeline.v216_public_knn_blend_runtime import _visual_descriptor


class PosePredictor:
    def __init__(self, checkpoint: Path, device: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = ActionPoseProjector(int(state["hidden"])).to(self.device)
        self.model.load_state_dict(state["model"]); self.model.eval().requires_grad_(False)
        self.action_mean = torch.from_numpy(state["action_mean"]).to(self.device)
        self.action_std = torch.from_numpy(state["action_std"]).to(self.device)
        self.target_mean = torch.from_numpy(state["target_mean"]).to(self.device)
        self.target_std = torch.from_numpy(state["target_std"]).to(self.device)

    @torch.inference_mode()
    def right(self, actions: np.ndarray) -> np.ndarray:
        value = torch.from_numpy(np.asarray(actions, np.float32)).to(self.device)
        arms = torch.ones(len(value), dtype=torch.long, device=self.device)
        normalized = normalize_action(value, self.action_mean, self.action_std, arms)
        output = self.model(normalized, arms)
        return denormalize_pose(output, self.target_mean, self.target_std, arms).float().cpu().numpy()


def mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(first.astype(np.float32) - second.astype(np.float32)).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.run / "phase_gate_report.json"
    details_output = args.run / "phase_gate_details.npz"
    if output.exists() or details_output.exists():
        raise FileExistsError("refusing overwrite")
    prereg = json.loads((args.run / "release_registration.json").read_text())
    split = json.loads(args.split.read_text())
    arms = {int(key): value for key, value in split["arm_by_episode"].items()}
    holdout = sorted(ep for ep in map(int, split["validation_episodes"]) if arms[ep] == "right")
    expected = prereg["fixed_acceptance"]["exact_holdout_episodes"]
    predictor = PosePredictor(args.pose, args.device)

    with np.load(args.library, allow_pickle=False) as lib:
        paths = lib["path"].astype(str)
        episode_id = lib["episode_id"].astype(np.int64)
        visual = lib["visual"].astype(np.float32)
    clean_rows = np.flatnonzero(episode_id >= 20000)
    row_episode = np.asarray([int(Path(paths[row]).stem.split("_")[0][7:]) - 20000 for row in clean_rows])
    row_start = np.asarray([int(Path(paths[row]).stem.split("_")[1]) for row in clean_rows])
    train_right = sorted(set(row_episode.tolist()))

    trajectories = {}
    for ep in sorted(set(train_right + holdout)):
        trajectory = load_robotwin_hdf5(args.dataset / f"episode{ep}.hdf5")
        trajectories[ep] = {
            "frames": trajectory.frames,
            "actions": trajectory.actions,
            "poses": predictor.right(trajectory.actions[:-1, 7:14]),
        }
    terminal_residual = []
    for ep in train_right:
        terminal = trajectories[ep]["poses"][-16:]
        terminal_residual.append(terminal - np.median(terminal, axis=0))
    residual = np.concatenate(terminal_residual)
    scale = np.maximum(np.sqrt(np.mean(residual ** 2, axis=0)), np.asarray([1.5] * 6 + [0.003], np.float32))

    phase_cart, phase_base, terminal_cart, terminal_base = [], [], [], []
    rgb_cart, rgb_base, rgb_copy, rgb_delta = [], [], [], []
    selected_rows = []
    anchor_radius = int(prereg["fixed_protocol"]["anchor_search_radius_rows"])
    future_radius = int(prereg["fixed_protocol"]["future_search_radius_rows"])
    for query_ep in holdout:
        query_length = len(trajectories[query_ep]["frames"])
        for path in sorted(args.windows.glob(f"episode{query_ep}_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                context = data["context_frames"].copy()
                history = data["history_actions"].astype(np.float32)
                future = data["future_actions"].astype(np.float32)
                target = data["target_frames"].copy()
                start = int(data["start"])
            query_pose = predictor.right(np.concatenate((history[-1:, 7:14], future[:, 7:14]), axis=0))
            descriptor = _visual_descriptor(context[-1])
            distance = ((visual[clean_rows] - descriptor) ** 2).mean(axis=1)
            local = int(np.argmin(distance)); ref_ep = int(row_episode[local]); base_start = int(row_start[local])
            ref = trajectories[ref_ep]; ref_pose = ref["poses"]; ref_length = len(ref["frames"])
            base_anchor = int(np.clip(base_start + 3, 0, len(ref_pose) - 1))
            lo = max(0, base_anchor - anchor_radius); hi = min(len(ref_pose), base_anchor + anchor_radius + 1)
            anchor_cost = np.mean(((ref_pose[lo:hi] - query_pose[0]) / scale) ** 2, axis=1)
            anchor = lo + int(np.argmin(anchor_cost))
            chosen = []
            cursor = anchor
            for pose in query_pose[1:]:
                stop = min(len(ref_pose), cursor + future_radius + 1)
                cost = np.mean(((ref_pose[cursor:stop] - pose) / scale) ** 2, axis=1)
                cursor += int(np.argmin(cost)); chosen.append(cursor)
            chosen = np.asarray(chosen, np.int64)
            truth_phase = np.arange(start + 5, start + 13, dtype=np.float32) / max(query_length - 1, 1)
            cart_phase = (chosen + 1).astype(np.float32) / max(ref_length - 1, 1)
            base_indices = np.clip(np.arange(base_start + 5, base_start + 13), 0, ref_length - 1)
            base_phase = base_indices.astype(np.float32) / max(ref_length - 1, 1)
            phase_cart.extend(np.abs(cart_phase - truth_phase)); phase_base.extend(np.abs(base_phase - truth_phase))
            terminal_cart.append(abs(float(cart_phase[-1] - truth_phase[-1])))
            terminal_base.append(abs(float(base_phase[-1] - truth_phase[-1])))
            cart_frames = ref["frames"][np.clip(chosen + 1, 0, ref_length - 1)]
            base_frames = ref["frames"][base_indices]
            anchor_frame = ref["frames"][min(anchor + 1, ref_length - 1)].astype(np.float32)
            delta = cart_frames.astype(np.float32) - anchor_frame[None]
            delta_frames = np.clip(np.rint(context[-1].astype(np.float32)[None] + delta), 0, 255).astype(np.uint8)
            rgb_cart.append(mae(cart_frames, target)); rgb_base.append(mae(base_frames, target))
            rgb_copy.append(mae(np.repeat(context[-1:,:,:,:], 8, axis=0), target)); rgb_delta.append(mae(delta_frames, target))
            selected_rows.append((query_ep, start, ref_ep, base_start, anchor, *chosen.tolist()))

    phase_cart = np.asarray(phase_cart); phase_base = np.asarray(phase_base)
    terminal_cart = np.asarray(terminal_cart); terminal_base = np.asarray(terminal_base)
    rgb_cart = np.asarray(rgb_cart); rgb_base = np.asarray(rgb_base); rgb_copy = np.asarray(rgb_copy); rgb_delta = np.asarray(rgb_delta)
    ratios = {
        "phase_mae": float(phase_cart.mean() / max(phase_base.mean(), 1e-9)),
        "terminal_phase_mae": float(terminal_cart.mean() / max(terminal_base.mean(), 1e-9)),
        "full_rgb_mae": float(rgb_cart.mean() / max(rgb_base.mean(), 1e-9)),
        "delta_rgb_mae": float(rgb_delta.mean() / max(rgb_copy.mean(), 1e-9)),
    }
    limits = prereg["fixed_acceptance"]
    checks = {
        "exact_holdout_episodes": holdout == expected,
        "all_outputs_finite": all(np.isfinite(value).all() for value in (phase_cart, phase_base, terminal_cart, terminal_base, rgb_cart, rgb_base, rgb_copy, rgb_delta)),
        "cartesian_phase_mae_ratio": ratios["phase_mae"] <= limits["cartesian_phase_mae_ratio_le"],
        "cartesian_terminal_phase_mae_ratio": ratios["terminal_phase_mae"] <= limits["cartesian_terminal_phase_mae_ratio_le"],
        "cartesian_rgb_mae_ratio": ratios["full_rgb_mae"] <= limits["cartesian_rgb_mae_ratio_le"],
        "cartesian_delta_rgb_mae_ratio": ratios["delta_rgb_mae"] <= limits["cartesian_delta_rgb_mae_ratio_le"],
    }
    report = {
        "format": "strict-track2-v374-cartesian-phase-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_episodes": holdout,
        "windows": len(rgb_cart),
        "frames": len(phase_cart),
        "metrics": {
            "cartesian_phase_mae": float(phase_cart.mean()), "baseline_phase_mae": float(phase_base.mean()),
            "cartesian_terminal_phase_mae": float(terminal_cart.mean()), "baseline_terminal_phase_mae": float(terminal_base.mean()),
            "cartesian_full_rgb_mae": float(rgb_cart.mean()), "baseline_full_rgb_mae": float(rgb_base.mean()),
            "cartesian_delta_rgb_mae": float(rgb_delta.mean()), "copy_last_rgb_mae": float(rgb_copy.mean()),
        },
        "ratios": ratios,
        "checks": checks,
        "passed": all(checks.values()),
        "authorizes_world_model_pilot": all(checks.values()),
        "guards": prereg["guards"],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(details_output, selected=np.asarray(selected_rows, np.int64), phase_cart=phase_cart, phase_base=phase_base,
                        terminal_cart=terminal_cart, terminal_base=terminal_base, rgb_cart=rgb_cart, rgb_base=rgb_base,
                        rgb_copy=rgb_copy, rgb_delta=rgb_delta, scale=scale)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
