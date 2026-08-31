#!/usr/bin/env python3
"""Freeze the v329 training-only interpolation sweep before running it."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v329_train_only_terminal_bridge_sweep_seed1499_20260822"
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
        "sweep": B / "pipeline/scripts/sweep_v329_train_only_terminal_bridge.py",
        "v328_runtime": B / "pipeline/wam_pipeline/v328_coherent_phase_trajectory_runtime.py",
        "v326_runtime": B / "pipeline/wam_pipeline/v326_blended_phase_terminal_runtime.py",
        "v328_failed_contract": J / "v328_v326_coherent_phase_trajectory_gate_seed1498_20260822/audit/contract_report.json",
        "v326_authorization": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/expensive_rl_authorization.json",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "phase_gate": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(paths["v328_failed_contract"].read_text()).get("passed") is not False:
        raise RuntimeError("v328 must be rejected before v329 tuning")
    if json.loads(paths["v326_authorization"].read_text()).get("launch_permission") is not True:
        raise RuntimeError("v326 parent is not authorized")
    payload = {
        "format": "strict-track2-v329-train-only-terminal-bridge-sweep-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "replace only the v326 repaired terminal with a fixed interpolation between "
            "the temporally coherent mirrored terminal and the v326 success terminal"
        ),
        "fixed_sweep": {
            "lambdas": [0.50, 0.65, 0.75, 0.85, 0.925, 1.0],
            "episodes": "all 15 declared public-training right-arm episodes",
            "window_stride": 8,
            "subset": "frozen v326 phase/action gate accepted and v315 failure signature absent",
            "frames_0_through_6": "bit-exact v317 mirrored trajectory",
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
        "candidate_use": (
            "selected lambda is training-derived only; it must be frozen into a new runtime "
            "and pass untouched public validation/local gates before any RL"
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
