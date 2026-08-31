#!/usr/bin/env python3
"""Audit the single frozen public-development v445 S1 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "strict-track2-v445-full-mirror-s1-offline-gate-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("inputs", "preregistration", "s0-audit", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    s0 = json.loads(args.s0_audit.read_text())
    with np.load(args.inputs, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    required = {
        "v169_frames", "mirror_frames", "shuffle_frames", "target_frames",
        "open_frames", "static_frames", "reverse_frames",
        "mirror_reward", "shuffle_reward", "open_reward", "static_reward", "reverse_reward",
        "format", "episode", "arm", "phase", "s1", "gate",
        "contract_mixed_left_exact", "contract_right_serial_batch_exact", "contract_batch_permutation_exact",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError(f"v445 S1 missing keys: {missing}")
    s1_mask = arrays["s1"].astype(bool)
    right = s1_mask & (arrays["arm"].astype(str) == "right")
    left = arrays["arm"].astype(str) == "left"
    baseline = arrays["v169_frames"][right].astype(np.float64)
    mirror = arrays["mirror_frames"][right].astype(np.float64)
    shuffle = arrays["shuffle_frames"][right].astype(np.float64)
    target = arrays["target_frames"][right].astype(np.float64)
    base_abs = np.abs(baseline - target)
    mirror_abs = np.abs(mirror - target)
    shuffle_abs = np.abs(shuffle - target)
    first8_ratio = float(mirror_abs[:, :8].mean() / base_abs[:, :8].mean())
    recursive_ratio = float(mirror_abs.mean() / base_abs.mean())
    chunk3_ratio = float(mirror_abs[:, 16:24].mean() / base_abs[:, 16:24].mean())
    chunk4_ratio = float(mirror_abs[:, 24:32].mean() / base_abs[:, 24:32].mean())
    true_shuffle_ratio = float(mirror_abs.mean() / shuffle_abs.mean())
    true_sample_mae = mirror_abs.mean(axis=(1, 2, 3, 4))
    shuffle_sample_mae = shuffle_abs.mean(axis=(1, 2, 3, 4))
    pairwise = float(np.mean(true_sample_mae < shuffle_sample_mae))
    rgb_margin = float(np.mean(shuffle_sample_mae - true_sample_mae))
    mirror_final_reward = arrays["mirror_reward"][right, -1].astype(np.float64)
    shuffle_final_reward = arrays["shuffle_reward"][right, -1].astype(np.float64)
    reward_pairwise = float(np.mean(mirror_final_reward > shuffle_final_reward))
    reward_margin = float(np.mean(mirror_final_reward - shuffle_final_reward))
    phase_metrics = {}
    phase_pass = {}
    for phase_name in ("early", "grasp", "postgrasp", "endpoint"):
        mask = right & (arrays["phase"].astype(str) == phase_name)
        phase_base = np.abs(arrays["v169_frames"][mask].astype(np.float64) - arrays["target_frames"][mask].astype(np.float64))
        phase_true = np.abs(arrays["mirror_frames"][mask].astype(np.float64) - arrays["target_frames"][mask].astype(np.float64))
        sample_base = phase_base.mean(axis=(1, 2, 3, 4))
        sample_true = phase_true.mean(axis=(1, 2, 3, 4))
        values = {
            "samples": int(mask.sum()),
            "first8_ratio": float(phase_true[:, :8].mean() / phase_base[:, :8].mean()),
            "recursive32_ratio": float(phase_true.mean() / phase_base.mean()),
            "rgb_wins": int(np.sum(sample_true < sample_base)),
        }
        phase_metrics[phase_name] = values
        phase_pass[phase_name] = values["samples"] == 8 and values["first8_ratio"] <= 1.0 and values["recursive32_ratio"] <= 1.0 and values["rgb_wins"] >= 4
    episode_wins = 0
    episode_metrics = {}
    for episode in sorted(set(arrays["episode"][right].astype(int).tolist())):
        mask = right & (arrays["episode"].astype(int) == episode)
        base_mae = float(np.abs(arrays["v169_frames"][mask].astype(np.float64) - arrays["target_frames"][mask].astype(np.float64)).mean())
        true_mae = float(np.abs(arrays["mirror_frames"][mask].astype(np.float64) - arrays["target_frames"][mask].astype(np.float64)).mean())
        episode_metrics[str(episode)] = {"v169_mae": base_mae, "mirror_mae": true_mae, "improved": true_mae < base_mae}
        episode_wins += int(true_mae < base_mae)
    counterfactual_metrics = {}
    counterfactual_pass = {}
    for name in ("open", "static", "reverse"):
        true_final = arrays["mirror_reward"][right, -1].astype(np.float64)
        cf_final = arrays[f"{name}_reward"][right, -1].astype(np.float64)
        difference = true_final - cf_final
        normalized = difference / (np.abs(true_final) + np.abs(cf_final) + 1e-6)
        effect = float(np.mean(np.abs(arrays["mirror_frames"][right].astype(np.float64) - arrays[f"{name}_frames"][right].astype(np.float64))))
        values = {
            "reward_pairwise": float(np.mean(difference > 0.0)),
            "normalized_reward_margin": float(np.mean(normalized)),
            "rgb_effect": effect,
        }
        counterfactual_metrics[name] = values
        counterfactual_pass[name] = values["reward_pairwise"] >= 0.75 and values["normalized_reward_margin"] >= 0.05 and values["rgb_effect"] > 1e-6
    checks = {
        "input_format": str(arrays["format"].item()) == "strict-track2-v445-full-mirror-s1-offline-inputs-v1",
        "preregistration_format": prereg.get("format") == "strict-track2-v445-full-mirror-preregistration-v1",
        "s0_passed": s0.get("format") == "strict-track2-v445-full-mirror-s0-static-contract-v1" and s0.get("passed") is True,
        "right_s1_exact32": int(right.sum()) == 32,
        "frame_contract": all(arrays[key].dtype == np.uint8 and arrays[key].ndim == 5 and arrays[key].shape[1] == 32 and arrays[key].shape[-1] == 3 for key in ("v169_frames", "mirror_frames", "shuffle_frames", "open_frames", "static_frames", "reverse_frames", "target_frames")),
        "reward_finite": all(np.isfinite(arrays[f"{name}_reward"]).all() for name in ("mirror", "shuffle", "open", "static", "reverse")),
        "right_gate_all_chunks": arrays["gate"][right].shape == (32, 4) and bool(np.all(arrays["gate"][right])),
        "left_gate_zero": bool(np.all(~arrays["gate"][left])) if left.any() else False,
        "left_bitexact_v169": np.array_equal(arrays["mirror_frames"][left], arrays["v169_frames"][left]) if left.any() else False,
        "mixed_batch_left_bitexact": bool(arrays["contract_mixed_left_exact"].item()),
        "right_serial_batch_bitexact": bool(arrays["contract_right_serial_batch_exact"].item()),
        "batch_permutation_bitexact": bool(arrays["contract_batch_permutation_exact"].item()),
        "first8_rgb_ratio": first8_ratio <= 0.998,
        "recursive32_rgb_ratio": recursive_ratio <= 1.0,
        "chunk3_rgb_ratio": chunk3_ratio <= 1.0,
        "chunk4_rgb_ratio": chunk4_ratio <= 1.0,
        "true_shuffle_rgb_ratio": true_shuffle_ratio <= 0.998,
        "true_shuffle_rgb_pairwise": pairwise >= 0.60,
        "true_shuffle_rgb_margin": rgb_margin > 0.0,
        "true_shuffle_final_reward_pairwise": reward_pairwise >= 0.60,
        "true_shuffle_final_reward_margin": reward_margin > 0.0,
        "all_phase_gates": all(phase_pass.values()),
        "right_episode_wins_min3": len(episode_metrics) == 4 and episode_wins >= 3,
        "open_reward_counterfactual": counterfactual_pass["open"],
        "static_reward_counterfactual": counterfactual_pass["static"],
        "reverse_reward_counterfactual": counterfactual_pass["reverse"],
    }
    passed = all(checks.values())
    report = {
        "format": FORMAT, "passed": passed, "checks": checks,
        "metrics": {
            "first8_rgb_mae_ratio": first8_ratio, "recursive32_rgb_mae_ratio": recursive_ratio,
            "chunk3_rgb_mae_ratio": chunk3_ratio, "chunk4_rgb_mae_ratio": chunk4_ratio,
            "true_to_shuffle_target_mae_ratio": true_shuffle_ratio,
            "true_better_shuffle_rgb_fraction": pairwise, "shuffle_minus_true_rgb_mae_margin": rgb_margin,
            "true_better_shuffle_final_reward_fraction": reward_pairwise,
            "true_minus_shuffle_final_reward_margin": reward_margin,
            "phase": phase_metrics, "episodes": episode_metrics, "right_episode_wins": episode_wins,
            "reward_counterfactual": counterfactual_metrics,
        },
        "sha256": {"inputs": sha256(args.inputs), "preregistration": sha256(args.preregistration), "s0_audit": sha256(args.s0_audit)},
        "guards": {"development_run_count": 1, "policy_updates": 0, "real_submission": False, "rl_authorized": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
