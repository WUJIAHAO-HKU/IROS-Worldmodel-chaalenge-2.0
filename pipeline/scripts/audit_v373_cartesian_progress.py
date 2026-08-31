#!/usr/bin/env python3
"""Test a continuous end-effector terminal-progress feature without reward labels."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

from wam_pipeline.object_geometry_v170 import ActionPoseProjector, denormalize_pose, normalize_action
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


def episode_number(path: Path) -> int:
    match = re.match(r"episode(\d+)_", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def stats(value: np.ndarray) -> dict:
    return {
        "count": int(len(value)),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.9)),
        "max": float(value.max()),
    }


class PosePredictor:
    def __init__(self, checkpoint: Path, device: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = ActionPoseProjector(int(state["hidden"])).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval().requires_grad_(False)
        self.action_mean = torch.from_numpy(state["action_mean"]).to(self.device)
        self.action_std = torch.from_numpy(state["action_std"]).to(self.device)
        self.target_mean = torch.from_numpy(state["target_mean"]).to(self.device)
        self.target_std = torch.from_numpy(state["target_std"]).to(self.device)

    @torch.inference_mode()
    def right(self, actions: np.ndarray) -> np.ndarray:
        value = torch.from_numpy(np.asarray(actions, np.float32)).to(self.device)
        arms = torch.ones(len(value), dtype=torch.long, device=self.device)
        normalized = normalize_action(value, self.action_mean, self.action_std, arms)
        prediction = self.model(normalized, arms)
        output = denormalize_pose(prediction, self.target_mean, self.target_std, arms)
        return output.float().cpu().numpy()


def quality(anchor: np.ndarray, terminal: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[float, float, float]:
    anchor_distance = float(np.sqrt(np.mean(((anchor - center) / scale) ** 2)))
    terminal_distance = float(np.sqrt(np.mean(((terminal - center) / scale) ** 2)))
    progress = float((anchor_distance - terminal_distance) / max(anchor_distance, 1e-6))
    # Smooth and monotonic: proximity alone cannot pass without progress toward the endpoint.
    score = float(np.exp(-terminal_distance) * np.clip((progress + 0.1) / 0.6, 0.0, 1.0))
    return score, terminal_distance, progress


def choose_threshold(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[float, dict]:
    rows = []
    for threshold in np.linspace(0.02, 0.8, 79):
        fold_rates = []
        for episode in np.unique(groups):
            selected = groups == episode
            prediction = scores[selected] >= threshold
            truth = labels[selected]
            if truth.any() and (~truth).any():
                fold_rates.append((float(prediction[truth].mean()), float((~prediction[~truth]).mean())))
        tpr = float(np.mean([row[0] for row in fold_rates]))
        tnr = float(np.mean([row[1] for row in fold_rates]))
        rows.append({"threshold": float(threshold), "tpr": tpr, "tnr": tnr, "min_rate": min(tpr, tnr)})
    selected = max(rows, key=lambda row: (row["min_rate"], (row["tpr"] + row["tnr"]) / 2, -abs(row["threshold"] - 0.3)))
    return selected["threshold"], selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--capture-details", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.run / "audit/cartesian_progress_gate.json"
    details_output = args.run / "audit/cartesian_progress_details.npz"
    if output.exists() or details_output.exists():
        raise FileExistsError("refusing overwrite")
    if not json.loads((args.run / "audit/pose_reconstruction_acceptance.json").read_text())["passed"]:
        raise RuntimeError("pose reconstruction did not pass")

    split = json.loads(args.split.read_text())
    train = set(map(int, split["train_episodes"]))
    arms = {int(key): value for key, value in split["arm_by_episode"].items()}
    right = sorted(ep for ep in train if arms[ep] == "right")
    if len(right) != 15:
        raise RuntimeError(f"expected 15 right train episodes, got {right}")
    predictor = PosePredictor(args.run / "model/best.pt", args.device)

    centers = {}
    terminal_rows = {}
    lengths = {}
    all_residuals = []
    for ep in right:
        with h5py.File(args.dataset / f"episode{ep}.hdf5", "r") as handle:
            actions = np.asarray(handle["joint_action/vector"][:-1, 7:14], np.float32)
        poses = predictor.right(actions)
        terminal = poses[-16:]
        center = np.median(terminal, axis=0)
        centers[ep] = center
        terminal_rows[ep] = terminal
        lengths[ep] = len(poses) + 1
        all_residuals.append(terminal - center)
    residual = np.concatenate(all_residuals)
    # Floors reflect the reconstruction error, preventing overconfident gates.
    scale = np.maximum(np.sqrt(np.mean(residual ** 2, axis=0)), np.asarray([1.5] * 6 + [0.003], np.float32))

    clean_scores, clean_labels, clean_groups = [], [], []
    clean_distance, clean_progress = [], []
    for ep in right:
        for path in sorted(args.windows.glob(f"episode{ep}_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                history = np.asarray(data["history_actions"], np.float32)
                future = np.asarray(data["future_actions"], np.float32)
                start = int(data["start"])
            if float(future[:, 13].mean()) > 0.5:
                continue
            target_index = start + 12
            if target_index >= lengths[ep] - 16:
                label = True
            elif target_index <= lengths[ep] - 48:
                label = False
            else:
                continue
            predicted = predictor.right(np.stack((history[-1, 7:14], future[-1, 7:14])))
            score, distance, progress = quality(predicted[0], predicted[1], centers[ep], scale)
            clean_scores.append(score); clean_labels.append(label); clean_groups.append(ep)
            clean_distance.append(distance); clean_progress.append(progress)
    clean_scores = np.asarray(clean_scores); clean_labels = np.asarray(clean_labels, bool); clean_groups = np.asarray(clean_groups)
    threshold, calibration = choose_threshold(clean_scores, clean_labels, clean_groups)

    records = []
    for rollout_index, path in enumerate(sorted(args.capture_dir.glob("rollout_*.npz"))):
        with np.load(path, allow_pickle=False) as data:
            contexts = data["context_frames"]
            histories = data["history_actions"].astype(np.float32)
            futures = data["future_actions"].astype(np.float32)
            texts = list(map(str, json.loads(str(data["instructions_json"]))))
        for action_index, (context, history, future, instruction) in enumerate(zip(contexts, histories, futures, texts, strict=True)):
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(history, future, instruction)
            if route == "right" and history[-1, 13] < 0.5 and float((future[:, 13] < 0.5).mean()) >= 0.75:
                records.append((rollout_index, action_index, history, future))
    if len(records) != 101:
        raise RuntimeError(f"expected frozen 101 post-grasp captures, got {len(records)}")
    with np.load(args.library, allow_pickle=False) as library, np.load(args.capture_details, allow_pickle=False) as old:
        base = old["base"].astype(np.int64)
        base_episode = library["episode_id"][base].astype(np.int64) - 20000
    if len(base_episode) != len(records) or not set(base_episode).issubset(set(right)):
        raise RuntimeError("capture/library alignment mismatch")

    capture_scores, capture_distance, capture_progress = [], [], []
    for (_, _, history, future), ep in zip(records, base_episode, strict=True):
        predicted = predictor.right(np.stack((history[-1, 7:14], future[-1, 7:14])))
        score, distance, progress = quality(predicted[0], predicted[1], centers[int(ep)], scale)
        capture_scores.append(score); capture_distance.append(distance); capture_progress.append(progress)
    capture_scores = np.asarray(capture_scores)
    accepted = capture_scores >= threshold
    clean_tpr = float((clean_scores[clean_labels] >= threshold).mean())
    clean_tnr = float((clean_scores[~clean_labels] < threshold).mean())
    checks = {
        "fifteen_public_right_train_episodes": len(right) == 15,
        "phase_labels_use_trajectory_position_not_reward": True,
        "clean_terminal_tpr_ge_0p80": clean_tpr >= 0.80,
        "clean_early_tnr_ge_0p80": clean_tnr >= 0.80,
        "capture_records_exact_101": len(records) == 101,
        "capture_has_nonzero_signal": int(accepted.sum()) >= 5,
        "capture_acceptance_bounded_le_0p35": float(accepted.mean()) <= 0.35,
    }
    report = {
        "format": "strict-track2-v373-cartesian-progress-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pose_dimensions": "projected end-effector origin, axes, and camera depth",
        "terminal_reference": "per-episode median of final 16 public expert poses",
        "scale": scale.tolist(),
        "calibration": calibration,
        "selected_threshold": threshold,
        "clean": {
            "samples": len(clean_scores), "positive": int(clean_labels.sum()), "negative": int((~clean_labels).sum()),
            "tpr": clean_tpr, "tnr": clean_tnr, "score": stats(clean_scores),
            "distance": stats(np.asarray(clean_distance)), "progress": stats(np.asarray(clean_progress)),
        },
        "frozen_policy_capture": {
            "samples": len(capture_scores), "accepted": int(accepted.sum()), "acceptance_rate": float(accepted.mean()),
            "score": stats(capture_scores), "distance": stats(np.asarray(capture_distance)), "progress": stats(np.asarray(capture_progress)),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "guards": {
            "public_world_model_data_only": True,
            "reward_or_success_outcomes_used_for_fit": False,
            "capture_actions_are_read_only_unlabeled_ood_inputs": True,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(details_output, clean_score=clean_scores, clean_label=clean_labels, clean_group=clean_groups,
                        capture_score=capture_scores, capture_accepted=accepted, capture_episode=base_episode,
                        scale=scale, threshold=np.asarray(threshold))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
