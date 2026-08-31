#!/usr/bin/env python3
"""Calibration/test analysis for bounded Cartesian refinement around the KNN phase."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.data import load_robotwin_hdf5
from wam_pipeline.object_geometry_v170 import ActionPoseProjector, denormalize_pose, normalize_action
from wam_pipeline.v216_public_knn_blend_runtime import _visual_descriptor


CALIBRATION_EPISODES = (6, 7)
TEST_EPISODES = (18, 22)
RADII = (2, 4, 8, 16)
PENALTIES = (0.0, 0.05, 0.2, 1.0)


class PosePredictor:
    def __init__(self, checkpoint: Path, device: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = ActionPoseProjector(int(state["hidden"])).to(self.device)
        self.model.load_state_dict(state["model"]); self.model.eval().requires_grad_(False)
        self.am = torch.from_numpy(state["action_mean"]).to(self.device)
        self.astd = torch.from_numpy(state["action_std"]).to(self.device)
        self.tm = torch.from_numpy(state["target_mean"]).to(self.device)
        self.tstd = torch.from_numpy(state["target_std"]).to(self.device)

    @torch.inference_mode()
    def right(self, actions: np.ndarray) -> np.ndarray:
        value = torch.from_numpy(np.asarray(actions, np.float32)).to(self.device)
        arm = torch.ones(len(value), dtype=torch.long, device=self.device)
        output = self.model(normalize_action(value, self.am, self.astd, arm), arm)
        return denormalize_pose(output, self.tm, self.tstd, arm).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing overwrite")
    split = json.loads(args.split.read_text())
    arms = {int(k): v for k, v in split["arm_by_episode"].items()}
    holdout = sorted(ep for ep in map(int, split["validation_episodes"]) if arms[ep] == "right")
    if holdout != sorted(CALIBRATION_EPISODES + TEST_EPISODES):
        raise RuntimeError(holdout)
    predictor = PosePredictor(args.pose, args.device)
    with np.load(args.library, allow_pickle=False) as lib:
        paths = lib["path"].astype(str); episode_id = lib["episode_id"].astype(np.int64); visual = lib["visual"].astype(np.float32)
    clean_rows = np.flatnonzero(episode_id >= 20000)
    row_ep = np.asarray([int(Path(paths[r]).stem.split("_")[0][7:]) - 20000 for r in clean_rows])
    row_start = np.asarray([int(Path(paths[r]).stem.split("_")[1]) for r in clean_rows])
    train_right = sorted(set(row_ep.tolist()))
    trajectories = {}
    for ep in sorted(set(train_right + holdout)):
        trajectory = load_robotwin_hdf5(args.dataset / f"episode{ep}.hdf5")
        trajectories[ep] = {"frames": trajectory.frames, "poses": predictor.right(trajectory.actions[:-1, 7:14])}
    residual = []
    for ep in train_right:
        terminal = trajectories[ep]["poses"][-16:]; residual.append(terminal - np.median(terminal, axis=0))
    scale = np.maximum(np.sqrt(np.mean(np.concatenate(residual) ** 2, axis=0)), np.asarray([1.5] * 6 + [0.003], np.float32))

    configs = [(radius, penalty) for radius in RADII for penalty in PENALTIES]
    sums = {split_name: {config: np.zeros(6, np.float64) for config in configs} for split_name in ("calibration", "test")}
    counts = {"calibration": 0, "test": 0}
    baseline = {"calibration": np.zeros(4, np.float64), "test": np.zeros(4, np.float64)}
    for query_ep in holdout:
        split_name = "calibration" if query_ep in CALIBRATION_EPISODES else "test"
        query_len = len(trajectories[query_ep]["frames"])
        for path in sorted(args.windows.glob(f"episode{query_ep}_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                context = data["context_frames"].copy(); history = data["history_actions"].astype(np.float32)
                future = data["future_actions"].astype(np.float32); target = data["target_frames"].copy(); start = int(data["start"])
            query_pose = predictor.right(future[:, 7:14])
            local = int(np.argmin(((visual[clean_rows] - _visual_descriptor(context[-1])) ** 2).mean(axis=1)))
            ref_ep = int(row_ep[local]); base_start = int(row_start[local]); ref = trajectories[ref_ep]
            ref_len = len(ref["frames"]); baseline_indices = np.clip(np.arange(base_start + 5, base_start + 13), 0, ref_len - 1)
            truth_phase = np.arange(start + 5, start + 13, dtype=np.float32) / max(query_len - 1, 1)
            baseline_phase = baseline_indices.astype(np.float32) / max(ref_len - 1, 1)
            base_frames = ref["frames"][baseline_indices]
            copy = np.repeat(context[-1:], 8, axis=0)
            baseline[split_name] += np.asarray([
                np.abs(baseline_phase - truth_phase).mean(), abs(float(baseline_phase[-1] - truth_phase[-1])),
                np.abs(base_frames.astype(np.float32) - target.astype(np.float32)).mean(),
                np.abs(copy.astype(np.float32) - target.astype(np.float32)).mean(),
            ])
            anchor_frame = ref["frames"][int(np.clip(base_start + 4, 0, ref_len - 1))].astype(np.float32)
            for config in configs:
                radius, penalty = config; selected = []; cursor = 0
                for horizon, (pose, center) in enumerate(zip(query_pose, baseline_indices, strict=True)):
                    lo = max(cursor, int(center) - radius, 0); hi = min(ref_len - 1, int(center) + radius)
                    pose_rows = ref["poses"][max(0, lo - 1):max(0, hi - 1) + 1]
                    frame_rows = np.arange(lo, hi + 1)
                    if len(pose_rows) != len(frame_rows):
                        # Frame zero has no preceding action pose; it is never reached in these windows.
                        frame_rows = frame_rows[-len(pose_rows):]
                    pose_cost = np.mean(((pose_rows - pose) / scale) ** 2, axis=1)
                    offset = (frame_rows - center) / max(radius, 1)
                    chosen = int(frame_rows[np.argmin(pose_cost + penalty * offset ** 2)])
                    selected.append(chosen); cursor = chosen
                selected = np.asarray(selected, np.int64)
                selected_phase = selected.astype(np.float32) / max(ref_len - 1, 1)
                frames = ref["frames"][selected]
                delta = frames.astype(np.float32) - anchor_frame[None]
                delta_frames = np.clip(np.rint(context[-1].astype(np.float32)[None] + delta), 0, 255)
                sums[split_name][config] += np.asarray([
                    np.abs(selected_phase - truth_phase).mean(), abs(float(selected_phase[-1] - truth_phase[-1])),
                    np.abs(frames.astype(np.float32) - target.astype(np.float32)).mean(),
                    np.abs(delta_frames - target.astype(np.float32)).mean(),
                    float(np.abs(selected - baseline_indices).mean()), float(np.max(np.abs(selected - baseline_indices))),
                ])
            counts[split_name] += 1

    results = {}
    for split_name in ("calibration", "test"):
        base = baseline[split_name] / counts[split_name]
        rows = []
        for config in configs:
            value = sums[split_name][config] / counts[split_name]
            rows.append({
                "radius": config[0], "penalty": config[1], "phase_mae": value[0], "terminal_phase_mae": value[1],
                "rgb_mae": value[2], "delta_rgb_mae": value[3], "mean_shift": value[4], "max_shift_mean": value[5],
                "phase_ratio": value[0] / base[0], "terminal_phase_ratio": value[1] / base[1],
                "rgb_ratio": value[2] / base[2], "delta_rgb_ratio": value[3] / base[3],
            })
        results[split_name] = {"windows": counts[split_name], "baseline": {"phase_mae": base[0], "terminal_phase_mae": base[1], "rgb_mae": base[2], "copy_rgb_mae": base[3]}, "rows": rows}
    eligible = [row for row in results["calibration"]["rows"] if row["phase_ratio"] <= 1.05 and row["terminal_phase_ratio"] <= 1.05]
    selected = min(eligible, key=lambda row: (row["delta_rgb_ratio"], row["rgb_ratio"], row["radius"], row["penalty"])) if eligible else None
    test_selected = None if selected is None else next(row for row in results["test"]["rows"] if row["radius"] == selected["radius"] and row["penalty"] == selected["penalty"])
    report = {
        "format": "strict-track2-v374-bounded-cartesian-sweep-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_inputs": list(CALIBRATION_EPISODES), "test_inputs": list(TEST_EPISODES),
        "selection_rule": "calibration phase and terminal ratios <=1.05, then minimum delta RGB ratio",
        "selected": selected, "test_selected": test_selected, "results": results,
        "guards": {"public_world_model_data_only": True, "reward_or_outcomes_used": False, "policy_modified": False, "real_submission": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "test_selected": test_selected, "calibration_windows": counts["calibration"], "test_windows": counts["test"]}, indent=2))


if __name__ == "__main__":
    main()
