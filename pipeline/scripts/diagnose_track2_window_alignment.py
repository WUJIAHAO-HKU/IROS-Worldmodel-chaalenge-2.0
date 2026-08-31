#!/usr/bin/env python3
"""Audit Track 2 arm labels against the official joint14 action blocks.

The audit is intentionally outcome-blind.  It uses only the recorded arm label,
the 14-D actions and (optionally) RGB frame deltas.  It never reads success when
deciding which arm is active or which windows should be sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LEFT_JOINTS = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_JOINTS = slice(7, 13)
RIGHT_GRIPPER = 13


def _episode_id(path: Path) -> int:
    return int(path.name.split("_")[0][7:])


def _mean_delta_by_dimension(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 2:
        raise ValueError(f"expected at least two joint14 actions, got {actions.shape}")
    return np.abs(np.diff(actions.astype(np.float64), axis=0)).mean(axis=0)


def _dominant_arm(left: float, right: float, margin: float) -> str:
    if right > left * margin:
        return "right"
    if left > right * margin:
        return "left"
    return "ambiguous"


def audit_windows(
    windows: Path,
    *,
    dominance_margin: float = 1.10,
    limit: int = 0,
    rgb_limit: int = 0,
) -> dict[str, object]:
    paths = sorted(windows.glob("episode*_*.npz"))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no Track 2 windows found under {windows}")
    if dominance_margin < 1.0:
        raise ValueError("dominance margin must be at least 1")

    raw_rows: list[dict[str, object]] = []
    action_total = np.zeros(14, dtype=np.float64)
    action_squared = np.zeros(14, dtype=np.float64)
    action_count = 0
    identity = hashlib.sha256()
    rgb_seen = 0
    for path in paths:
        identity.update(path.name.encode("utf-8"))
        identity.update(b"\n")
        with np.load(path, allow_pickle=False) as data:
            required = {"history_actions", "future_actions", "arm_right"}
            if not required.issubset(data.files):
                raise ValueError(f"{path}: missing {sorted(required.difference(data.files))}")
            history = np.asarray(data["history_actions"], dtype=np.float64)
            future = np.asarray(data["future_actions"], dtype=np.float64)
            if history.shape != (4, 14) or future.shape != (8, 14):
                raise ValueError(f"{path}: invalid action shapes {history.shape}, {future.shape}")
            arm_value = np.asarray(data["arm_right"])
            if arm_value.shape != ():
                raise ValueError(f"{path}: arm_right must be scalar")
            actions = np.concatenate((history, future), axis=0)
            delta = _mean_delta_by_dimension(actions)
            action_total += actions.sum(axis=0)
            action_squared += np.square(actions).sum(axis=0)
            action_count += len(actions)
            row: dict[str, object] = {
                "file": path.name,
                "episode": _episode_id(path),
                "label": "right" if bool(arm_value) else "left",
                "delta_by_dimension": delta.tolist(),
                "left_gripper_delta": float(delta[LEFT_GRIPPER]),
                "right_gripper_delta": float(delta[RIGHT_GRIPPER]),
            }
            if rgb_seen < rgb_limit and {"context_frames", "target_frames"}.issubset(data.files):
                context = np.asarray(data["context_frames"], dtype=np.float32)
                target = np.asarray(data["target_frames"], dtype=np.float32)
                previous = np.concatenate((context[-1:], target[:-1]), axis=0)
                frame_delta = np.abs(target - previous)
                midpoint = frame_delta.shape[2] // 2
                row["rgb_left_half_delta"] = float(frame_delta[:, :, :midpoint].mean())
                row["rgb_right_half_delta"] = float(frame_delta[:, :, midpoint:].mean())
                rgb_seen += 1
        raw_rows.append(row)

    mean = action_total / action_count
    std = np.sqrt(np.maximum(action_squared / action_count - np.square(mean), 1e-8))
    episode_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        normalized = np.asarray(row.pop("delta_by_dimension"), dtype=np.float64) / std
        left = float(normalized[LEFT_JOINTS].mean())
        right = float(normalized[RIGHT_JOINTS].mean())
        dominant = _dominant_arm(left, right, dominance_margin)
        row.update(
            {
                "normalized_left_joint_motion": left,
                "normalized_right_joint_motion": right,
                "right_to_left_motion_ratio": right / max(left, 1e-12),
                "dominant_action_arm": dominant,
                "label_action_agree": dominant == row["label"],
            }
        )
        episode_rows[int(row["episode"])].append(row)

    episodes = []
    for episode, rows in sorted(episode_rows.items()):
        labels = {str(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"episode {episode}: arm label changes between windows")
        left = float(np.mean([float(row["normalized_left_joint_motion"]) for row in rows]))
        right = float(np.mean([float(row["normalized_right_joint_motion"]) for row in rows]))
        dominant = _dominant_arm(left, right, dominance_margin)
        label = labels.pop()
        episodes.append(
            {
                "episode": episode,
                "label": label,
                "window_count": len(rows),
                "normalized_left_joint_motion": left,
                "normalized_right_joint_motion": right,
                "right_to_left_motion_ratio": right / max(left, 1e-12),
                "dominant_action_arm": dominant,
                "label_action_agree": dominant == label,
            }
        )

    def summarize(rows: list[dict[str, object]], label: str | None = None) -> dict[str, object]:
        selected = rows if label is None else [row for row in rows if row["label"] == label]
        decisive = [row for row in selected if row["dominant_action_arm"] != "ambiguous"]
        return {
            "count": len(selected),
            "decisive_count": len(decisive),
            "label_action_agreement_all": float(np.mean([bool(row["label_action_agree"]) for row in selected])) if selected else None,
            "label_action_agreement_decisive": float(np.mean([bool(row["label_action_agree"]) for row in decisive])) if decisive else None,
            "dominant_left": sum(row["dominant_action_arm"] == "left" for row in selected),
            "dominant_right": sum(row["dominant_action_arm"] == "right" for row in selected),
            "ambiguous": sum(row["dominant_action_arm"] == "ambiguous" for row in selected),
            "normalized_left_joint_motion_mean": float(np.mean([float(row["normalized_left_joint_motion"]) for row in selected])) if selected else None,
            "normalized_right_joint_motion_mean": float(np.mean([float(row["normalized_right_joint_motion"]) for row in selected])) if selected else None,
        }

    return {
        "format": "strict-track2-joint14-arm-audit-v2",
        "windows": str(windows.resolve()),
        "window_identity_sha256": identity.hexdigest(),
        "official_joint14_layout": "left_joint[0:6], left_gripper[6], right_joint[7:13], right_gripper[13]",
        "outcome_labels_used_for_arm_inference": False,
        "dominance_margin": dominance_margin,
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "window_summary": {
            "all": summarize(raw_rows),
            "label_left": summarize(raw_rows, "left"),
            "label_right": summarize(raw_rows, "right"),
        },
        "episode_summary": {
            "all": summarize(episodes),
            "label_left": summarize(episodes, "left"),
            "label_right": summarize(episodes, "right"),
        },
        "rgb_audited_windows": rgb_seen,
        "episodes": episodes,
        "rows": raw_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 audits every window")
    parser.add_argument("--rgb-limit", type=int, default=0)
    parser.add_argument("--dominance-margin", type=float, default=1.10)
    parser.add_argument("--minimum-episode-agreement", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_episode_agreement <= 1.0:
        raise SystemExit("minimum episode agreement must be in [0, 1]")
    report = audit_windows(
        args.windows,
        dominance_margin=args.dominance_margin,
        limit=args.limit,
        rgb_limit=args.rgb_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "format": report["format"],
        "window_summary": report["window_summary"],
        "episode_summary": report["episode_summary"],
        "rgb_audited_windows": report["rgb_audited_windows"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    agreement = report["episode_summary"]["all"]["label_action_agreement_all"]
    if agreement is not None and agreement < args.minimum_episode_agreement:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
