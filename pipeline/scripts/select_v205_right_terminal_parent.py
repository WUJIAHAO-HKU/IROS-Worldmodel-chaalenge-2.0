#!/usr/bin/env python3
"""Apply the preregistered right-arm holdout gates and freeze one expert."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = BASE / "artifacts/strict_track2_official_20260810"
TAG = "v205_v202_public_right_terminal_multichunk_seed1405"
RUN = JOINT / TAG
REG = OFFICIAL / "run_registry" / TAG / "preregistration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def right_visual_metrics(candidate_cache: Path) -> dict[str, float | int]:
    with np.load(RUN / "audit/baseline/public_demo_holdout.npz", allow_pickle=False) as values:
        target = values["target"].astype(np.float32)
        baseline = values["baseline"].astype(np.float32)
        arm_right = values["arm_right"].astype(bool)
        paths = values["path"].astype(str).tolist()
    with np.load(candidate_cache, allow_pickle=False) as values:
        candidate = values["candidate"].astype(np.float32)
        candidate_paths = values["path"].astype(str).tolist()
    if candidate_paths != paths:
        raise ValueError("candidate and baseline cache paths differ")
    if int(arm_right.sum()) < 12:
        raise ValueError("right holdout has too few recursive sequences")
    base = float(np.mean(np.abs(baseline[arm_right] - target[arm_right])))
    cand = float(np.mean(np.abs(candidate[arm_right] - target[arm_right])))
    return {
        "right_sequences": int(arm_right.sum()),
        "baseline_rgb_mae": base,
        "candidate_rgb_mae": cand,
        "regression_percent": 100.0 * (cand - base) / max(base, 1e-12),
    }


def ratio(report: dict, group: str, metric: str) -> float:
    row = report["groups"][group]
    return float(row["candidate"][metric] / max(row["baseline"][metric], 1e-12))


def main() -> None:
    prereg = json.loads(REG.read_text())
    fixed = prereg["fixed_training"]
    gates = prereg["fixed_audit"]["gates"]
    rows = []
    for step in fixed["checkpoint_steps"]:
        root = RUN / "audit" / f"step{step:03d}"
        reward = json.loads((root / "public_demo_reward.json").read_text())
        visual = right_visual_metrics(root / "public_demo_candidate.npz")
        reward_ratio = ratio(
            reward, "right", "reward_mae_normalized_by_gt_std"
        )
        terminal_ratio = ratio(reward, "right", "terminal_gain_mae")
        high_reward_ratio = ratio(
            reward, "right_high_motion", "reward_mae_normalized_by_gt_std"
        )
        selection_metric = float(np.mean([reward_ratio, terminal_ratio]))
        checks = {
            "right_visual": visual["regression_percent"]
            <= gates["right_visual_mae_max_regression_percent"],
            "right_reward": reward_ratio <= gates["right_reward_mae_ratio_max"],
            "right_terminal": terminal_ratio
            <= gates["right_terminal_gain_mae_ratio_max"],
            "right_high_motion_reward": high_reward_ratio
            <= gates["right_high_motion_reward_mae_ratio_max"],
            "must_improve": (
                selection_metric < 1.0
                if gates["must_improve_right_selection_metric"]
                else True
            ),
        }
        rows.append(
            {
                "step": step,
                "selection_metric": selection_metric,
                "right_reward_mae_ratio": reward_ratio,
                "right_terminal_gain_mae_ratio": terminal_ratio,
                "right_high_motion_reward_mae_ratio": high_reward_ratio,
                "right_visual": visual,
                "checks": checks,
                "failed_checks": [name for name, value in checks.items() if not value],
                "passed": all(checks.values()),
            }
        )

    eligible = [row for row in rows if row["passed"]]
    selected = min(
        eligible, key=lambda row: (row["selection_metric"], row["step"])
    ) if eligible else None
    destination = RUN / "selected_right_expert"
    if destination.exists():
        raise SystemExit("refusing to overwrite selected_right_expert")
    if selected is not None:
        source = RUN / "checkpoints" / f"checkpoint_step_{selected['step']:06d}"
        destination.mkdir()
        for name in (
            "model.pt",
            "action_normalization.npz",
            "track2_autoregressive_unet_config.npz",
            "training_manifest.json",
        ):
            shutil.copy2(source / name, destination / name)
        selected_sha = sha256(destination / "model.pt")
    else:
        source = None
        selected_sha = None

    report = {
        "format": "strict-track2-v205-public-right-terminal-parent-audit-v1",
        "passed": selected is not None,
        "selected": selected,
        "candidates": rows,
        "preregistration": str(REG),
        "preregistration_sha256": sha256(REG),
        "selected_source": str(source) if source else None,
        "selected_right_expert": str(destination) if selected else None,
        "selected_model_sha256": selected_sha,
        "selection_inputs": (
            "untouched public 10-episode holdout only; no development policy, "
            "hidden, final-128, or submission result"
        ),
        "left_expert_unchanged": str(
            JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
        ),
        "real_submission_performed": False,
    }
    output = RUN / "audit/right_terminal_parent_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    marker = RUN / ("V205_RIGHT_PARENT_ACCEPTED" if selected else "V205_RIGHT_PARENT_REJECTED")
    marker.touch()
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "selected_step": selected["step"] if selected else None,
                "selected_metric": selected["selection_metric"] if selected else None,
                "candidates": [
                    {
                        "step": row["step"],
                        "metric": row["selection_metric"],
                        "passed": row["passed"],
                        "failed": row["failed_checks"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
