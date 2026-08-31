#!/usr/bin/env python3
"""Freeze the v330 public-train proxy sweep before measuring rewards."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v330_train_phase_proxy_terminal_bridge_seed1500_20260822"
RUN = J / N
REG = O / "run_registry" / N


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    paths = {
        "sweep": B / "pipeline/scripts/sweep_v330_train_phase_proxy_terminal_bridge.py",
        "v329_helper": B / "pipeline/scripts/sweep_v329_train_only_terminal_bridge.py",
        "v328_runtime": B / "pipeline/wam_pipeline/v328_coherent_phase_trajectory_runtime.py",
        "v329_failed_log": J / "v329_train_only_terminal_bridge_sweep_seed1499_20260822/audit/sweep.log",
        "v326_authorization": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/expensive_rl_authorization.json",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "phase_gate": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if "too few phase-qualified train requests: 0" not in paths["v329_failed_log"].read_text():
        raise RuntimeError("v329 zero-coverage failure is not present")
    payload = {
        "format": "strict-track2-v330-train-phase-proxy-terminal-bridge-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "fixed_sweep": {
            "lambdas": [0.50, 0.65, 0.75, 0.85, 0.925, 1.0],
            "episodes": "all 15 declared public-training right-arm episodes",
            "window_stride": 8,
            "expected_proxy_windows": 46,
            "proxy_gates": (
                "right post-grasp, action probability >=0.99, no frozen failure signature, "
                "and frozen public phase ready; baseline_alpha==0 omitted only in train tuning"
            ),
            "repaired_terminal": "the exact v326 alpha-0.90 success-terminal construction",
            "frames_0_through_6": "v317 mirrored trajectory",
            "frame_7": "(1-lambda)*mirrored_terminal + lambda*v326_repaired_terminal",
        },
        "selection_rule": {
            "terminal_reward_mean_min": 0.90,
            "terminal_reward_hit_rate_at_0p9_min": 0.85,
            "next_context_rgb_mae_ratio_vs_v326_max": 1.02,
            "next_context_temporal_delta_error_ratio_vs_v326_max": 0.95,
            "terminal_reward_error_ratio_vs_v326_max": 1.10,
            "objective": "among eligible lambdas minimize temporal ratio, then RGB ratio, then prefer larger lambda",
            "fail_closed_if_none_eligible": True,
        },
        "next_step": (
            "freeze the single selected lambda in a new runtime and evaluate it once on the "
            "untouched public validation/local gates; this sweep cannot authorize RL"
        ),
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "public_training_episodes_only": True,
            "public_holdout_or_evaluation_outcomes": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "preregistration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
