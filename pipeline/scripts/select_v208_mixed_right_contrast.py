#!/usr/bin/env python3
"""Apply the preregistered v208 two-holdout gates and freeze one right expert."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFF = BASE / "artifacts/strict_track2_official_20260810"
TAG = "v208_v205_mixed_right_gripper_contrast_long32_seed1407"
RUN = JOINT / TAG
REG = OFF / "run_registry" / TAG / "preregistration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visual_ratio(baseline_cache: Path, candidate_cache: Path) -> tuple[float, dict[str, float | int]]:
    with np.load(baseline_cache, allow_pickle=False) as values:
        target = values["target"].astype(np.float32)
        baseline = values["baseline"].astype(np.float32)
        arm_right = values["arm_right"].astype(bool)
        paths = values["path"].astype(str).tolist()
    with np.load(candidate_cache, allow_pickle=False) as values:
        candidate = values["candidate"].astype(np.float32)
        candidate_paths = values["path"].astype(str).tolist()
    if paths != candidate_paths or int(arm_right.sum()) < 8:
        raise ValueError("candidate cache is not aligned or has too few right sequences")
    base = float(np.mean(np.abs(baseline[arm_right] - target[arm_right])))
    cand = float(np.mean(np.abs(candidate[arm_right] - target[arm_right])))
    return cand / max(base, 1e-12), {
        "right_sequences": int(arm_right.sum()),
        "baseline_mae": base,
        "candidate_mae": cand,
    }


def reward_ratio(report: dict, metric: str) -> float:
    row = report["groups"]["right"]
    return float(row["candidate"][metric] / max(row["baseline"][metric], 1e-12))


def main() -> None:
    prereg = json.loads(REG.read_text())
    gates = prereg["fixed_audit"]["gates"]
    rows = []
    for step in prereg["fixed_training"]["checkpoint_steps"]:
        root = RUN / "audit" / f"step{step:04d}"
        public_report = json.loads((root / "public_success_reward.json").read_text())
        failure_report = json.loads((root / "public_failure_reward.json").read_text())
        public_visual, public_visual_detail = visual_ratio(
            RUN / "audit/baseline/public_success_holdout.npz",
            root / "public_success_candidate.npz",
        )
        failure_visual, failure_visual_detail = visual_ratio(
            RUN / "audit/baseline/public_failure_holdout.npz",
            root / "public_failure_candidate.npz",
        )
        ratios = {
            "public_right_visual": public_visual,
            "public_right_reward": reward_ratio(public_report, "reward_mae_normalized_by_gt_std"),
            "public_right_terminal": reward_ratio(public_report, "terminal_gain_mae"),
            "failure_right_visual": failure_visual,
            "failure_right_reward": reward_ratio(failure_report, "reward_mae_normalized_by_gt_std"),
            "failure_right_terminal": reward_ratio(failure_report, "terminal_gain_mae"),
        }
        selection_metric = float(np.mean(list(ratios.values())))
        checks = {
            "public_right_visual": ratios["public_right_visual"] <= gates["public_right_visual_mae_ratio_max"],
            "public_right_reward": ratios["public_right_reward"] <= gates["public_right_reward_mae_ratio_max"],
            "public_right_terminal": ratios["public_right_terminal"] <= gates["public_right_terminal_gain_mae_ratio_max"],
            "failure_right_visual": ratios["failure_right_visual"] <= gates["failure_right_visual_mae_ratio_max"],
            "failure_right_reward": ratios["failure_right_reward"] <= gates["failure_right_reward_mae_ratio_max"],
            "failure_right_terminal": ratios["failure_right_terminal"] <= gates["failure_right_terminal_gain_mae_ratio_max"],
            "mean_ratio_must_improve": selection_metric < 1.0 if gates["mean_ratio_must_improve"] else True,
        }
        rows.append({
            "step": step,
            "selection_metric": selection_metric,
            "ratios": ratios,
            "public_visual": public_visual_detail,
            "failure_visual": failure_visual_detail,
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "passed": all(checks.values()),
        })

    eligible = [row for row in rows if row["passed"]]
    selected = min(eligible, key=lambda row: (row["selection_metric"], row["step"])) if eligible else None
    destination = RUN / "selected_right_expert"
    if destination.exists():
        raise SystemExit("refusing to overwrite v208 selected_right_expert")
    source = None
    selected_sha = None
    if selected:
        source = RUN / "checkpoints" / f"checkpoint_step_{selected['step']:06d}"
        destination.mkdir()
        for name in ("model.pt", "action_normalization.npz", "track2_autoregressive_unet_config.npz", "training_manifest.json"):
            shutil.copy2(source / name, destination / name)
        selected_sha = sha256(destination / "model.pt")

    report = {
        "format": "strict-track2-v208-mixed-right-gripper-contrast-audit-v1",
        "passed": selected is not None,
        "selected": selected,
        "candidates": rows,
        "preregistration": str(REG),
        "preregistration_sha256": sha256(REG),
        "selected_source": str(source) if source else None,
        "selected_right_expert": str(destination) if selected else None,
        "selected_model_sha256": selected_sha,
        "selection_inputs": "public training demonstrations and public on-policy holdouts only",
        "hidden_or_final_evaluation_data": False,
        "policy_action_injection": False,
        "real_submission_performed": False,
    }
    output = RUN / "audit/v208_right_contrast_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (RUN / ("V208_RIGHT_PARENT_ACCEPTED" if selected else "V208_RIGHT_PARENT_REJECTED")).touch()
    print(json.dumps({"passed": report["passed"], "selected": selected, "candidates": rows}, indent=2))


if __name__ == "__main__":
    main()
