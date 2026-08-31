#!/usr/bin/env python3
"""Apply the frozen v202 dual-holdout gates and materialize one candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = BASE / "artifacts/strict_track2_official_20260810"
RUN = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402"
REG = OFFICIAL / "run_registry/v202_v201_public_terminal_reward_calibration_seed1402/preregistration.json"
PARENT = JOINT / "v201_v196_public_long16_sourceweighted_seed1401/model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visual_metrics(baseline_cache: Path, candidate_cache: Path) -> dict:
    with np.load(baseline_cache, allow_pickle=False) as values:
        target = values["target"].astype(np.float32)
        baseline = values["baseline"].astype(np.float32)
        arm_right = values["arm_right"].astype(bool)
        context = values["context_last"].astype(np.float32)
        paths = values["path"].astype(str).tolist()
    with np.load(candidate_cache, allow_pickle=False) as values:
        candidate = values["candidate"].astype(np.float32)
        candidate_paths = values["path"].astype(str).tolist()
    if paths != candidate_paths:
        raise ValueError("candidate and baseline path order differs")
    previous = np.concatenate((context[:, None], target[:, :-1]), axis=1)
    motion = np.mean(np.abs(target - previous) / 255.0, axis=(1, 2, 3, 4))
    high = motion >= 0.03
    groups = {
        "overall": np.ones(len(paths), dtype=bool),
        "left": ~arm_right,
        "right": arm_right,
        "left_low_motion": (~arm_right) & (~high),
        "left_high_motion": (~arm_right) & high,
        "right_low_motion": arm_right & (~high),
        "right_high_motion": arm_right & high,
    }
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        base = float(np.mean(np.abs(baseline[mask] - target[mask])))
        cand = float(np.mean(np.abs(candidate[mask] - target[mask])))
        result[name] = {
            "windows": int(mask.sum()),
            "baseline_rgb_mae": base,
            "candidate_rgb_mae": cand,
            "regression_percent": 100.0 * (cand - base) / max(base, 1e-12),
        }
    return result


def reward_ratios(report: dict) -> dict:
    result = {}
    for group in ("overall", "right", "right_high_motion"):
        row = report["groups"][group]
        base, candidate = row["baseline"], row["candidate"]
        result[group] = {
            "reward_mae_ratio": candidate["reward_mae_normalized_by_gt_std"]
            / max(base["reward_mae_normalized_by_gt_std"], 1e-12),
            "terminal_gain_mae_ratio": candidate["terminal_gain_mae"]
            / max(base["terminal_gain_mae"], 1e-12),
            "delta_mae_ratio": candidate["delta_mae"] / max(base["delta_mae"], 1e-12),
        }
    return result


def main() -> None:
    prereg = json.loads(REG.read_text())
    gates = prereg["fixed_audit"]["gates"]
    rows = []
    for step in prereg["fixed_training"]["checkpoint_steps"]:
        tag = f"step{step:03d}"
        root = RUN / "audit" / tag
        datasets = {}
        selection_terms = []
        checks = {}
        for name in ("public_demo", "public_onpolicy"):
            reward = json.loads((root / f"{name}_reward.json").read_text())
            ratios = reward_ratios(reward)
            visual = visual_metrics(
                RUN / "audit/baseline" / f"{name}_holdout.npz",
                root / f"{name}_candidate.npz",
            )
            datasets[name] = {"reward_ratios": ratios, "visual": visual}
            selection_terms.extend([
                ratios["overall"]["reward_mae_ratio"],
                ratios["overall"]["terminal_gain_mae_ratio"],
            ])
            checks[f"{name}_visual"] = (
                visual["overall"]["regression_percent"]
                <= gates["visual_mae_max_regression_percent"]
            )
            checks[f"{name}_overall_reward"] = (
                ratios["overall"]["reward_mae_ratio"]
                <= 1.0 + gates["overall_reward_mae_max_regression_percent"] / 100.0
            )
            checks[f"{name}_overall_terminal"] = (
                ratios["overall"]["terminal_gain_mae_ratio"]
                <= 1.0 + gates["overall_terminal_gain_mae_max_regression_percent"] / 100.0
            )
            checks[f"{name}_right_terminal"] = (
                ratios["right"]["terminal_gain_mae_ratio"]
                <= 1.0 + gates["right_terminal_gain_mae_max_regression_percent"] / 100.0
            )
            checks[f"{name}_right_high_reward"] = (
                ratios["right_high_motion"]["reward_mae_ratio"]
                <= 1.0 + gates["right_high_motion_reward_mae_max_regression_percent"] / 100.0
            )
        metric = float(np.mean(selection_terms))
        checks["must_improve"] = metric < 1.0
        rows.append({
            "step": step,
            "selection_metric": metric,
            "passed": all(checks.values()),
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "datasets": datasets,
        })

    eligible = [row for row in rows if row["passed"]]
    selected = min(eligible, key=lambda row: (row["selection_metric"], row["step"])) if eligible else None
    source = (
        RUN / "checkpoints" / f"checkpoint_step_{selected['step']:06d}"
        if selected is not None else PARENT
    )
    destination = RUN / "selected_model"
    if destination.exists():
        raise SystemExit("refusing to overwrite selected_model")
    destination.mkdir()
    for name in ("model.pt", "action_normalization.npz", "track2_autoregressive_unet_config.npz", "training_manifest.json"):
        shutil.copy2(source / name, destination / name)

    report = {
        "format": "strict-track2-v202-public-terminal-reward-calibration-audit-v1",
        "passed": selected is not None,
        "selected": selected,
        "fallback_parent_retained": selected is None,
        "candidates": rows,
        "preregistration": str(REG),
        "preregistration_sha256": sha256(REG),
        "selected_source": str(source),
        "selected_model": str(destination),
        "selected_model_sha256": sha256(destination / "model.pt"),
        "selection_inputs": "two episode-disjoint public-only holdouts; no hidden/final evaluation result",
        "official_reward_model_modified": False,
        "real_submission_performed": False,
    }
    output = RUN / "audit/p2_reward_calibration_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "selected_step": selected["step"] if selected else 0,
        "selection_metric": selected["selection_metric"] if selected else 1.0,
        "candidate_summary": [
            {"step": row["step"], "metric": row["selection_metric"], "passed": row["passed"], "failed": row["failed_checks"]}
            for row in rows
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
