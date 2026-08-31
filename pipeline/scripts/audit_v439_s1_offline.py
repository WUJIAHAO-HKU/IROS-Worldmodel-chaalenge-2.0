#!/usr/bin/env python3
"""Offline three-way S1 gate for v439.

The inference runner is intentionally separate.  It must write an immutable
NPZ containing v169, the learned baseline, hybrid, shuffled-action hybrid and
real target predictions/rewards.  This auditor never calls a model and never
passes reward, seed, or request id into a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REQUIRED = {
    "v169_frames", "baseline_frames", "hybrid_frames", "shuffle_frames",
    "target_frames", "v169_reward", "baseline_reward", "hybrid_reward",
    "target_reward", "episode", "arm", "phase", "gate", "swap_gate", "s1",
}
HOLDOUT10 = {2, 5, 6, 7, 9, 16, 18, 22, 39, 48}
PROTECTED_IN_CHUNK = (0, 1, 6, 7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean())


def ratio(value: float, reference: float) -> float:
    return float(value / max(reference, 1e-12))


def frame_metrics(frames: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "first8_rgb_mae": mae(frames[:, :8], target[:, :8]),
        "recursive32_rgb_mae": mae(frames, target),
    }


def reward_metrics(reward: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "reward_prediction_mae": float(np.abs(reward - target).mean()),
        "endpoint_reward_prediction_mae": float(np.abs(reward[:, -1] - target[:, -1]).mean()),
        "final_reward_mean": float(reward[:, -1].mean()),
    }


def exact(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def protected_indices(horizon: int) -> list[int]:
    if horizon % 8:
        raise ValueError(f"horizon must be a multiple of eight, got {horizon}")
    return [offset + item for offset in range(0, horizon, 8) for item in PROTECTED_IN_CHUNK]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--static-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    static = json.loads(args.static_report.read_text(encoding="utf-8"))
    if static.get("format") != "strict-track2-v439-static-contract-audit-v1":
        raise RuntimeError("wrong v439 static report")

    with np.load(args.npz, allow_pickle=False) as values:
        missing = sorted(REQUIRED - set(values.files))
        if missing:
            raise RuntimeError(f"S1 input is missing arrays: {missing}")
        arrays = {key: np.asarray(values[key]) for key in REQUIRED}

    v169 = arrays["v169_frames"]
    baseline = arrays["baseline_frames"]
    hybrid = arrays["hybrid_frames"]
    shuffle = arrays["shuffle_frames"]
    target = arrays["target_frames"]
    frame_shape_ok = (
        v169.shape == baseline.shape == hybrid.shape == shuffle.shape == target.shape
        and v169.ndim == 5 and v169.shape[1] == 32 and v169.shape[-1] == 3
        and all(item.dtype == np.uint8 for item in (v169, baseline, hybrid, shuffle, target))
    )
    if not frame_shape_ok:
        raise RuntimeError("expected five aligned uint8 [N,32,H,W,3] frame arrays")
    count = len(v169)
    reward_shape = (count, 32)
    if any(arrays[key].shape != reward_shape for key in ("v169_reward", "baseline_reward", "hybrid_reward", "target_reward")):
        raise RuntimeError(f"reward arrays must be {reward_shape}")
    if any(arrays[key].shape != (count,) for key in ("episode", "arm", "phase", "s1")):
        raise RuntimeError("episode/arm/phase/s1 must be one-dimensional and aligned")
    if any(arrays[key].shape != (count, 4) for key in ("gate", "swap_gate")):
        raise RuntimeError("gate/swap_gate must be request-level [N,4] arrays")

    arm = arrays["arm"].astype("U")
    phase = arrays["phase"].astype("U")
    gate = arrays["gate"].astype(bool)
    swap_gate = arrays["swap_gate"].astype(bool)
    s1 = arrays["s1"].astype(bool)
    episodes = {int(value) for value in arrays["episode"].tolist()}
    left = arm == "left"
    right = arm == "right"
    protected = protected_indices(hybrid.shape[1])
    if not s1.any() or not np.all(right[s1]):
        raise RuntimeError("S1 metric rows must be a nonempty right-arm-only subset")

    metrics = {}
    for name, frames, rewards in (
        ("v169", v169, arrays["v169_reward"]),
        ("baseline", baseline, arrays["baseline_reward"]),
        ("hybrid", hybrid, arrays["hybrid_reward"]),
    ):
        metrics[name] = {
            **frame_metrics(frames[s1], target[s1]),
            **reward_metrics(
                rewards[s1].astype(np.float64),
                arrays["target_reward"][s1].astype(np.float64),
            ),
        }
    comparisons = {
        key: ratio(metrics["hybrid"][key], metrics["v169"][key])
        for key in (
            "first8_rgb_mae", "recursive32_rgb_mae", "reward_prediction_mae",
            "endpoint_reward_prediction_mae",
        )
    }
    comparisons["final_reward_hybrid_over_v169"] = ratio(
        metrics["hybrid"]["final_reward_mean"], metrics["v169"]["final_reward_mean"]
    )
    true_action_mae = metrics["hybrid"]["recursive32_rgb_mae"]
    shuffle_action_mae = mae(shuffle[s1], target[s1])
    true_over_shuffle = ratio(true_action_mae, shuffle_action_mae)
    intervention_fraction = float(gate[s1].mean())
    applied_delta = hybrid.astype(np.int16) - v169.astype(np.int16)
    g0_chunk_exact = True
    for sample_index in range(count):
        for chunk_index in range(4):
            if not gate[sample_index, chunk_index]:
                begin = 8 * chunk_index
                g0_chunk_exact &= exact(
                    hybrid[sample_index, begin : begin + 8],
                    v169[sample_index, begin : begin + 8],
                )

    checks = {
        "static_contract_passed": static.get("passed") is True,
        "samples_nonempty": count > 0,
        "s1_exact_32_samples": int(s1.sum()) == 32,
        "s1_phase_8_each": all(
            int(np.sum(s1 & (phase == name))) == 8
            for name in ("early", "grasp", "postgrasp", "endpoint")
        ),
        "fixed_public_holdout_dev_only": episodes <= HOLDOUT10 and bool(episodes),
        "s1_exact_public_right_dev_episodes": {
            int(value) for value in arrays["episode"][s1].tolist()
        } == {6, 7, 18, 22},
        "both_arms_covered": bool(left.any() and right.any()),
        "left_bitexact_v169": bool(left.any() and exact(hybrid[left], v169[left])),
        "g0_request_chunk_bitexact_v169": bool((~gate).any() and g0_chunk_exact),
        "t1_t2_t7_t8_each_chunk_bitexact_v169": exact(hybrid[:, protected], v169[:, protected]),
        # Static audit proves the pre-alpha residual clip constant.  This
        # dynamic bound independently proves no emitted uint8 pixel moved by
        # more than that cap.
        "emitted_uint8_delta_bounded_by_cap8": bool(
            np.max(np.abs(applied_delta), initial=0) <= 8
        ),
        "action_swap_changes_gate": bool(np.any(gate != swap_gate)),
        "swap_disables_at_least_one_original_intervention": bool(np.any(gate & ~swap_gate)),
        "first8_hybrid_over_v169_le_0p998": comparisons["first8_rgb_mae"] <= 0.998,
        "recursive32_hybrid_over_v169_le_1p002": comparisons["recursive32_rgb_mae"] <= 1.002,
        "reward_mae_nonregression": comparisons["reward_prediction_mae"] <= 1.0,
        "endpoint_reward_mae_nonregression": comparisons["endpoint_reward_prediction_mae"] <= 1.0,
        "final_reward_mean_ge_0p95_v169": metrics["hybrid"]["final_reward_mean"]
        >= 0.95 * metrics["v169"]["final_reward_mean"],
        "true_action_vs_shuffle_target_mae_improves_0p2pct": true_over_shuffle <= 0.998,
        "right_intervention_fraction_ge_0p20": intervention_fraction >= 0.20,
        "right_intervention_fraction_le_0p80": intervention_fraction <= 0.80,
        "all_rewards_finite": all(
            np.isfinite(arrays[key]).all()
            for key in ("v169_reward", "baseline_reward", "hybrid_reward", "target_reward")
        ),
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v439-s1-offline-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "coverage": {
            "samples": count,
            "s1_samples": int(s1.sum()),
            "episodes": sorted(episodes),
            "left": int(left.sum()),
            "right": int(right.sum()),
            "requests": int(gate.size),
            "gate_active_requests": int(gate.sum()),
            "right_intervention_fraction": intervention_fraction,
            "action_swap_gate_change_fraction": float((gate != swap_gate).mean()),
        },
        "three_way_metrics": metrics,
        "hybrid_over_v169": comparisons,
        "action_causality": {
            "true_action_target_recursive32_mae": true_action_mae,
            "shuffle_action_target_recursive32_mae": shuffle_action_mae,
            "true_over_shuffle": true_over_shuffle,
        },
        "checks": checks,
        "decision": "may prepare zero-update trace only" if passed else "reject before rollout/RL",
        "evidence_sha256": {
            "s1_npz": sha256(args.npz),
            "static_report": sha256(args.static_report),
        },
        "guards": {
            "models_loaded_by_auditor": False,
            "gate_receives_reward": False,
            "gate_receives_seed": False,
            "gate_receives_request_id": False,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
